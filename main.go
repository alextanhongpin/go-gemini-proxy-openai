package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"log/slog"

	"github.com/alextanhongpin/core/http/httpdump"
	"github.com/google/generative-ai-go/genai"
	openai "github.com/sashabaranov/go-openai"
	"google.golang.org/api/option"
)

type contextKey string

const systemPrompt = "I will ask you a question. Please answer it."

// Gemini roles.
const (
	roleUser  = "user"
	roleModel = "model"
)

// Errors.
var ErrInvalidParams = errors.New("invalid parameters")

var (

	// Global logger.
	logger *slog.Logger

	// Store all the clients per api key.
	clients sync.Map

	// ApiKey context key.
	apiKeyContextKey contextKey = "api_key"
)

type openaiAdapter interface {
	ChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
}

func init() {
	// Initialize the new slog handler
	logger = slog.New(slog.NewTextHandler(os.Stdout, nil))
}

func main() {
	defer func() {
		// Terminate all clients before exiting.
		clients.Range(func(key, val any) bool {
			_ = val.(*genai.Client).Close()

			// Continue iterating over the map.
			return true
		})
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/chat/completions", openaiHandler{new(gemini)}.ChatCompletion)
	mux.HandleFunc("/health", health)
	mux.HandleFunc("/", catchAll)

	logger.Info("Listening on port *:8080. press ctrl + c to cancel")
	panic(http.ListenAndServe(":8080", mux))
}

func catchAll(w http.ResponseWriter, r *http.Request) {
	logger.Error("not found", slog.Any("path", r.RequestURI))
	w.WriteHeader(http.StatusNotFound)
	w.Write([]byte("404 - Not Found"))
}

func health(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

var finishReasons = map[genai.FinishReason]openai.FinishReason{
	genai.FinishReasonUnspecified: openai.FinishReasonNull,
	genai.FinishReasonStop:        openai.FinishReasonStop,
	genai.FinishReasonMaxTokens:   openai.FinishReasonLength,
	genai.FinishReasonSafety:      openai.FinishReasonContentFilter,
	genai.FinishReasonRecitation:  openai.FinishReasonContentFilter,
	genai.FinishReasonOther:       openai.FinishReasonNull,
}

var geminiRolesByOpenAIRoles = map[string]string{
	"system":    roleUser,
	"assistant": roleModel,
	"user":      roleUser,
}

type openaiHandler struct {
	adapter openaiAdapter
}

func (h openaiHandler) ChatCompletion(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	// Let the client handle authorization error.
	apiKey := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	ctx = context.WithValue(ctx, apiKeyContextKey, apiKey)

	b, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	r.Body = io.NopCloser(bytes.NewReader(b))

	var req openai.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	r.Body = io.NopCloser(bytes.NewReader(b))

	// Log all request.
	logger.Info("request", slog.Any("body", req))

	resp, err := h.adapter.ChatCompletion(ctx, req)
	if err != nil {
		logger.Error("chat completion failed", slog.String("error", err.Error()))
		http.Error(w, err.Error(), http.StatusUnprocessableEntity)

		b, err = httpdump.DumpRequest(r)
		if err != nil {
			logger.Error("failed to dump http", slog.String("error", err.Error()))
			return
		}
		date := time.Now().Format(time.DateTime)
		if err := WriteIfNotExists(fmt.Sprintf("./data/request-%s.txt", date), b); err != nil {
			logger.Error("failed to write request", slog.String("error", err.Error()))
		}
		return
	}

	// Log all response.
	logger.Info("response", slog.Any("body", resp))

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	p, err := json.MarshalIndent(resp, "", " ")
	if err != nil {
		logger.Error("failed to marshal resp", slog.String("error", err.Error()))
		return
	}

	b, err = httpdump.DumpRequest(r)
	if err != nil {
		logger.Error("failed to dump http", slog.String("error", err.Error()))
		return
	}
	b = append(b, []byte("\n\n")...)
	b = append(b, p...)
	date := time.Now().Format(time.DateTime)
	if err := WriteIfNotExists(fmt.Sprintf("./data/request-%s.txt", date), b); err != nil {
		logger.Error("failed to write request", slog.String("error", err.Error()))
	}
}

type gemini struct {
	openaiAdapter
}

func (g *gemini) ChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	msgs, err := mergeOpenAIMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	contents, err := openaiToGenaiContents(msgs)
	if err != nil {
		return nil, err
	}

	model, err := g.loadOrStoreModel(ctx, req, isMultiModal(contents))
	if err != nil {
		return nil, err
	}

	// Chat messages must have roles alternating between 'user' and 'model'.
	sc := model.StartChat()

	// The first message must be from role `user`
	contents = fixContentRoleOrder(contents)

	var last *genai.Content
	last, contents = contents[len(contents)-1], contents[:len(contents)-1]
	sc.History = contents

	// The send message must be from role `user`.
	resp, err := sc.SendMessage(ctx, last.Parts...)
	if err != nil {
		return nil, err
	}

	return genaiToOpenaiResponse(resp)
}

func (g *gemini) createClient(ctx context.Context) (*genai.Client, error) {
	apiKey := ctx.Value(apiKeyContextKey).(string)
	client, ok := clients.Load(apiKey)
	if !ok {
		g, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
		if err != nil {
			return nil, err
		}

		c, loaded := clients.LoadOrStore(apiKey, g)
		if loaded {
			client = c
		} else {
			client = g
		}
	}

	return client.(*genai.Client), nil
}

// mergeOpenAIMessages merge the messages with the same role. This ensures that there
// are no consecutive messages with the same role.
// For example, if the messages are:
// [
//
//	{role: "user", content: "hello"},
//	{role: "user", content: "world"},
//	{role: "assistant", content: "hi"},
//	{role: "assistant", content: "there"},
//
// ]
// The result will be:
// [
//
//	{role: "user", content: "hello\nworld"},
//	{role: "assistant", content: "hi\nthere"},
//
// ]
func mergeOpenAIMessages(msgs []openai.ChatCompletionMessage) ([]openai.ChatCompletionMessage, error) {
	var prevRole string
	var res []openai.ChatCompletionMessage

	for _, msg := range msgs {
		role, ok := geminiRolesByOpenAIRoles[msg.Role]
		if !ok {
			return nil, fmt.Errorf("%w: failed to map openai role to gemini role: role=%q", ErrInvalidParams, msg.Role)
		}

		if role == prevRole {
			// Merge the content if the roles are similar.
			prevMsg := res[len(res)-1]

			lmc := prevMsg.MultiContent
			lc := prevMsg.Content
			rmc := msg.MultiContent
			rc := msg.Content

			lok := len(lmc) > 0
			rok := len(rmc) > 0

			switch {
			case lok && rok:
				lmc = append(lmc, rmc...)
			case lok && !rok:
				lmc = append(lmc, openai.ChatMessagePart{
					Type: openai.ChatMessagePartTypeText,
					Text: rc,
				})
			case !lok && rok:
				lmc = append([]openai.ChatMessagePart{
					{
						Type: openai.ChatMessagePartTypeText,
						Text: lc,
					},
				}, rmc...)
				lc = ""
			case !lok && !rok:
				lc = strings.Join([]string{lc, rc}, "\n")
			}

			prevMsg.MultiContent = lmc
			prevMsg.Content = lc
			res[len(res)-1] = prevMsg
		} else {
			prevRole = role
			res = append(res, msg)
		}
	}

	return res, nil
}

func (g *gemini) loadOrStoreModel(ctx context.Context, req openai.ChatCompletionRequest, isMultiModal bool) (*genai.GenerativeModel, error) {
	client, err := g.createClient(ctx)
	if err != nil {
		return nil, err
	}

	var model *genai.GenerativeModel
	if isMultiModal {
		model = client.GenerativeModel("gemini-pro-vision")
	} else {
		model = client.GenerativeModel("gemini-pro")
	}

	var (
		// Gemini only supports 1 candidate for now.
		candidateCount  = int32(1)
		maxOutputTokens = int32(req.MaxTokens)
		stopSequences   = req.Stop
		temperature     = req.Temperature
		topP            = req.TopP
	)

	model.SetCandidateCount(candidateCount)
	model.SetMaxOutputTokens(maxOutputTokens)
	model.SetTemperature(temperature)
	model.SetTopP(topP)
	model.StopSequences = stopSequences

	// Don't set if it is 0.
	if topP == 0 {
		model.TopP = nil
	}

	if maxOutputTokens == 0 {
		model.MaxOutputTokens = nil
	}

	logger.Info("parameters",
		slog.Int("candidate_count", int(candidateCount)),
		slog.Int("max_output_tokens", int(maxOutputTokens)),
		slog.String("stop_sequences", strings.Join(stopSequences, " ")),
		slog.Float64("temperature", float64(temperature)),
		slog.Float64("top_p", float64(topP)),
		slog.Bool("isMultiModal", isMultiModal),
	)

	return model, nil
}

func isMultiModal(contents []*genai.Content) bool {
	for _, c := range contents {
		for _, p := range c.Parts {
			_, ok := p.(genai.Blob)
			if ok {
				return ok
			}
		}
	}

	return false
}

func genaiToOpenaiResponse(resp *genai.GenerateContentResponse) (*openai.ChatCompletionResponse, error) {
	var res openai.ChatCompletionResponse
	res.Choices = make([]openai.ChatCompletionChoice, len(resp.Candidates))

	var tokens int
	for i, c := range resp.Candidates {
		role := IF(c.Content.Role == roleModel, "assistant", "user")
		idx := int(c.Index)
		content := mergeText(c.Content.Parts)
		finishReason := finishReasons[c.FinishReason]
		tokenCount := int(c.TokenCount)
		if finishReason == "" {
			return nil, fmt.Errorf("%w: failed to map gemini finish reason to openai finish reason: finish_reason=%q", ErrInvalidParams, c.FinishReason)
		}

		res.Choices[i] = openai.ChatCompletionChoice{
			Index: idx,
			Message: openai.ChatCompletionMessage{
				Role:    role,
				Content: content,
			},
			FinishReason: finishReason,
		}

		tokens += tokenCount
	}

	res.Usage.CompletionTokens = tokens

	return &res, nil
}

func IF[T any](ok bool, a, b T) T {
	if ok {
		return a
	}

	return b
}

func mergeText(in []genai.Part) string {
	t, ok := in[0].(genai.Text)
	if !ok {
		panic("part is not text")
	}

	texts := []string{string(t)}

	var j int
	for j = 1; j < len(in); j++ {
		t, ok := in[j].(genai.Text)
		if !ok {
			panic("part is not text")
		}
		texts = append(texts, string(t))
	}

	return strings.Join(texts, "")
}

// WriteIfNotExists writes the file to
// the designated location, only if it
// does not exists.
// Creates the folder too.
func WriteIfNotExists(name string, body []byte) error {
	f, err := os.OpenFile(name, os.O_RDONLY, 0644)
	if errors.Is(err, os.ErrNotExist) {
		dir := filepath.Dir(name)

		if err := os.MkdirAll(dir, 0700); err != nil && !os.IsExist(err) {
			return err
		} // Create your file

		return os.WriteFile(name, body, 0644)
	}
	if err != nil {
		return err
	}

	defer f.Close()

	return nil
}

func decodeBase64Image(b64 string) (mimeType string, blob []byte, err error) {
	lhs, rhs, ok := strings.Cut(b64, ";")
	if !ok {
		err = fmt.Errorf("%w: invalid image url", ErrInvalidParams)
		return
	}

	mimeType = strings.ReplaceAll(lhs, "data:", "")
	b64Image := strings.ReplaceAll(rhs, "base64,", "")

	blob, err = base64.StdEncoding.DecodeString(b64Image)
	return
}

func openaiToGenaiContent(msg openai.ChatCompletionMessage) (*genai.Content, error) {
	r := msg.Role
	c := msg.Content
	mc := msg.MultiContent

	var parts []genai.Part
	if len(mc) > 0 {
		parts = make([]genai.Part, len(mc))

		for j, content := range mc {
			ctype := content.Type
			ctext := content.Text

			switch ctype {
			case openai.ChatMessagePartTypeText:
				parts[j] = genai.Text(ctext)

			case openai.ChatMessagePartTypeImageURL:
				b64img := content.ImageURL.URL

				mimeType, blob, err := decodeBase64Image(b64img)
				if err != nil {
					return nil, err
				}

				format := strings.TrimPrefix(mimeType, "image/")
				parts[j] = genai.ImageData(format, blob)
			}
		}
	} else {
		parts = append(parts, genai.Text(c))
	}

	return &genai.Content{
		Role:  geminiRolesByOpenAIRoles[r],
		Parts: parts,
	}, nil
}

func openaiToGenaiContents(msgs []openai.ChatCompletionMessage) ([]*genai.Content, error) {
	contents := make([]*genai.Content, len(msgs))

	for i, msg := range msgs {
		content, err := openaiToGenaiContent(msg)
		if err != nil {
			return nil, err
		}
		contents[i] = content
	}

	return contents, nil
}

func fixContentRoleOrder(contents []*genai.Content) []*genai.Content {
	if contents[len(contents)-1].Role != roleUser {
		panic("last message must be from user")
	}

	if contents[0].Role == roleUser {
		return contents
	}

	return append([]*genai.Content{{
		Role:  roleUser,
		Parts: []genai.Part{genai.Text(systemPrompt)},
	}}, contents...)
}
