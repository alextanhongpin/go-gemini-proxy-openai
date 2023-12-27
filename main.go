package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"log/slog"

	"github.com/alextanhongpin/core/http/httpdump"
	"github.com/google/generative-ai-go/genai"
	openai "github.com/sashabaranov/go-openai"
	"google.golang.org/api/option"
)

const (
	roleUser  = "user"
	roleModel = "model"
)

var ErrInvalidParams = errors.New("invalid parameters")

var logger *slog.Logger

type openaiAdapter interface {
	ChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
}

func init() {
	if os.Getenv("GOOGLE_API_KEY") == "" {
		log.Fatal("GOOGLE_API_KEY environment variable must be set")
	}

	// Initialize the new slog handler
	logger = slog.New(slog.NewTextHandler(os.Stdout, nil))
}

// TODO: Cleanup the implementation.
func main() {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GOOGLE_API_KEY")))
	if err != nil {
		panic(err)
	}
	defer client.Close()

	g := newGemini(ctx, os.Getenv("GOOGLE_API_KEY"))

	mux := http.NewServeMux()
	mux.HandleFunc("/chat/completions", openaiHandler{g}.ChatCompletion)
	mux.HandleFunc("/health", health)
	mux.HandleFunc("/", catchAll)

	logger.Info("Listening on port *:8080. press ctrl + c to cancel")
	panic(http.ListenAndServe(":8080", mux))
}

func catchAll(w http.ResponseWriter, r *http.Request) {
	logger.Info("request", slog.Any("path", r.RequestURI))
	w.WriteHeader(http.StatusNotFound)
	w.Write([]byte("404 - Not Found"))
}

func health(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ok"))
}

var finishReasons = map[genai.FinishReason]openai.FinishReason{
	genai.FinishReasonUnspecified: openai.FinishReasonNull,
	genai.FinishReasonStop:        openai.FinishReasonStop,
	genai.FinishReasonMaxTokens:   openai.FinishReasonLength,
	genai.FinishReasonSafety:      openai.FinishReasonContentFilter,
	genai.FinishReasonRecitation:  openai.FinishReasonContentFilter,
	genai.FinishReasonOther:       openai.FinishReasonNull,
}

var toGeminiRole = map[string]string{
	"system":    roleModel,
	"assistant": roleModel,
	"user":      roleUser,
}

type openaiHandler struct {
	adapter openaiAdapter
}

func (h openaiHandler) ChatCompletion(w http.ResponseWriter, r *http.Request) {
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

	ctx := r.Context()
	resp, err := h.adapter.ChatCompletion(ctx, req)
	if err != nil {
		logger.Error("chat completion failed", slog.String("error", err.Error()))
		http.Error(w, err.Error(), http.StatusUnprocessableEntity)
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
	client *genai.Client
}

func newGemini(ctx context.Context, apiKey string) *gemini {
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		panic(err)
	}
	return &gemini{client: client}
}

func (g *gemini) ChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	model := g.client.GenerativeModel("gemini-pro")

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
	)

	var lastRole string
	var msgs []openai.ChatCompletionMessage
	for _, msg := range req.Messages {
		role, ok := toGeminiRole[msg.Role]
		if !ok {
			return nil, fmt.Errorf("%w: failed to map openai role to gemini role: role=%q", ErrInvalidParams, msg.Role)
		}

		if role == lastRole {
			// Merge the content if the roles are similar.
			lastMsg := msgs[len(msgs)-1]
			lastMsg.Content += " " + msg.Content
			msgs[len(msgs)-1] = lastMsg
		} else {
			lastRole = msg.Role
			msgs = append(msgs, msg)
		}
	}

	contents := make([]*genai.Content, len(msgs))
	for i, msg := range msgs {
		role := msg.Role
		content := msg.Content

		contents[i] = &genai.Content{
			Role:  toGeminiRole[role],
			Parts: []genai.Part{genai.Text(content)},
		}
	}

	// Chat messages must have roles alternating between 'user' and 'model'.
	sc := model.StartChat()

	// The first message must be from role `user`
	if contents[0].Role != roleUser {
		contents = append([]*genai.Content{{
			Role:  roleUser,
			Parts: []genai.Part{genai.Text("I will ask you a question. Please answer it.")},
		}}, contents...)
	}

	var last *genai.Content
	last, contents = contents[len(contents)-1], contents[:len(contents)-1]
	sc.History = contents

	for i, c := range contents {
		fmt.Println("------------------>", i+1, c.Role, c.Parts)
	}
	fmt.Println(mergeText(last.Parts))

	// The send message must be from role `user`.
	resp, err := sc.SendMessage(ctx, genai.Text(mergeText(last.Parts)))
	if err != nil {
		return nil, err
	}

	return g.toGeminiResponse(resp)
}

func (g *gemini) toGeminiResponse(resp *genai.GenerateContentResponse) (*openai.ChatCompletionResponse, error) {
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

func mergeTexts(in []genai.Part) []genai.Part {
	var out []genai.Part
	i := 0
	for i < len(in) {
		if t, ok := in[i].(genai.Text); ok {
			texts := []string{string(t)}
			var j int
			for j = i + 1; j < len(in); j++ {
				if t, ok := in[j].(genai.Text); ok {
					texts = append(texts, string(t))
				} else {
					break
				}
			}
			// j is just after the last Text.
			out = append(out, genai.Text(strings.Join(texts, "")))
			i = j
		} else {
			out = append(out, in[i])
			i++
		}
	}
	return out
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
