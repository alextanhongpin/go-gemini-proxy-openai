package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"log/slog"

	"github.com/google/generative-ai-go/genai"
	openai "github.com/sashabaranov/go-openai"
	"google.golang.org/api/option"
)

const roleUser = "user"
const roleModel = "model"

var logger *slog.Logger
var googleAPIKey string

func init() {
	googleAPIKey = os.Getenv("GOOGLE_API_KEY")
	if googleAPIKey == "" {
		log.Fatal("GOOGLE_API_KEY environment variable must be set")
	}

	// Initialize the new slog handler
	logger = slog.New(slog.NewTextHandler(os.Stdout, nil))
}

// TODO: Cleanup the implementation.
func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/chat/completions", CreateChatCompletionHandler)
	mux.HandleFunc("/healthz", health)
	// Add a catch all route
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		logger.Info("request", slog.Any("path", r.RequestURI))
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte("404 - Not Found"))
	})
	logger.Info("Listening on port *:8080. press ctrl + c to cancel")
	http.ListenAndServe(":8080", mux)
}

func health(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ok"))
}

func CreateChatCompletionHandler(w http.ResponseWriter, r *http.Request) {
	var req openai.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// TODO: Map to gemini
	// Gemini golang library.
	logger.Info("request", slog.Any("body", req))

	ctx := r.Context()
	client, err := genai.NewClient(ctx, option.WithAPIKey(googleAPIKey))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()
	//model := client.GenerativeModel("gemini-pro")

	resp, err := openaiToGeminiChatCompletion(ctx, req, client)
	if err != nil {
		logger.Error("error calling api", slog.String("error", err.Error()))
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	logger.Info("response", slog.Any("body", resp))

	var res *openai.ChatCompletionResponse
	res, err = geminiToOpenAIChatCompletion(resp)
	if err := json.NewEncoder(w).Encode(res); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
}

func geminiToOpenAIChatCompletion(resp *genai.GenerateContentResponse) (*openai.ChatCompletionResponse, error) {
	var res openai.ChatCompletionResponse
	res.Choices = make([]openai.ChatCompletionChoice, len(resp.Candidates))
	var tokens int
	for i, c := range resp.Candidates {
		var role string
		switch c.Content.Role {
		case "model":
			role = "assistant"
		case "user":
			role = "user"
		default:
			log.Fatalf("unsupported role: %q", c.Content.Role)
		}
		res.Choices[i] = openai.ChatCompletionChoice{
			Index: int(c.Index),
			Message: openai.ChatCompletionMessage{
				Role:    role,
				Content: string(mergeTexts(c.Content.Parts)[0].(genai.Text)),
			},
			FinishReason: finishReasons[c.FinishReason],
		}
		tokens += int(c.TokenCount)
	}
	res.Usage.CompletionTokens = tokens
	return &res, nil
}

func openaiToGeminiChatCompletion(ctx context.Context, req openai.ChatCompletionRequest, client *genai.Client) (*genai.GenerateContentResponse, error) {
	model := client.GenerativeModel("gemini-pro")
	model.SetCandidateCount(max(int32(req.N), 1))
	model.StopSequences = req.Stop

	if n := req.MaxTokens; n > 0 {
		model.SetMaxOutputTokens(int32(n))
	} else {
		model.SetMaxOutputTokens(4000)
	}

	if req.Temperature > 0 {
		model.SetTemperature(req.Temperature)
	}

	if req.TopP > 0 {
		model.SetTopP(req.TopP)
	}

	//model.SafetySettings = []*genai.SafetySetting{
	//{
	//Category:  genai.HarmCategoryDangerousContent,
	//Threshold: genai.HarmBlockLowAndAbove,
	//},
	//{
	//Category:  genai.HarmCategoryHarassment,
	//Threshold: genai.HarmBlockMediumAndAbove,
	//},
	//}

	logger.Info("max tokens",
		slog.Int("candidate_count", req.N),
		slog.Int("max", req.MaxTokens),
		slog.Float64("temperature", float64(req.Temperature)),
		slog.Float64("top_p", float64(req.TopP)),
		slog.Any("stop_sequences", req.Stop),
	)

	logger.Info("got number of messages", slog.Int("count", len(req.Messages)))
	sc := model.StartChat()
	var history []*genai.Content
	for i, msg := range req.Messages[:len(req.Messages)-1] {
		role, ok := toGeminiRole[msg.Role]
		if !ok {
			log.Fatalf("unsupported role: %q", msg.Role)
		}
		c := &genai.Content{
			Role:  role,
			Parts: []genai.Part{genai.Text(msg.Content)},
		}
		history = append(history, c)
		logger.Info("history", slog.Int("index", i), slog.Any("content", fmt.Sprintf("%+v", c)))
	}

	// Chat must be alternating roles between user and model.
	// The first message must be from role `user`
	if history[0].Role != roleUser {
		sc.History = append(sc.History, &genai.Content{
			Role:  roleUser,
			Parts: []genai.Part{genai.Text("hi, how are you?")},
		})
	}

	for len(history) > 0 {
		var h *genai.Content
		h, history = history[0], history[1:]

		last := sc.History[len(sc.History)-1]
		if last.Role == h.Role {
			hist := sc.History[len(sc.History)-1]
			hist.Parts = append(hist.Parts, h.Parts...)
			sc.History[len(sc.History)-1] = hist
		} else {
			sc.History = append(sc.History, h)
		}
	}

	// The last message must be from role `model`.
	if sc.History[len(sc.History)-1].Role != roleModel {
		sc.History = append(sc.History, &genai.Content{
			Role:  roleModel,
			Parts: []genai.Part{genai.Text("thinking...")},
		})
	}

	// The send message must be from role `user`.
	last := req.Messages[len(req.Messages)-1]
	resp, err := sc.SendMessage(ctx, genai.Text(last.Content))
	if err != nil {
		return nil, fmt.Errorf("failed to send message: %w", err)
	}

	return resp, nil
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
