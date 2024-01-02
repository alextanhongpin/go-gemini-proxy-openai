package goai

import (
	"context"
	"encoding/base64"
	"errors"
	"strings"
	"sync"
	"time"

	"log/slog"

	"github.com/google/generative-ai-go/genai"
	"github.com/google/uuid"
	openai "github.com/sashabaranov/go-openai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

type contextKey string

var (
	// ApiKey context key.
	apiKeyContextKey contextKey = "api_key"
)

func AuthContext(ctx context.Context, apiKey string) context.Context {
	return context.WithValue(ctx, apiKeyContextKey, apiKey)
}

type openaiClient interface {
	ChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
	ChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (chan openai.ChatCompletionStreamResponse, error)
}

type Adapter struct {
	openaiClient
	clients sync.Map
	logger  *slog.Logger
}

var _ openaiClient = (*Adapter)(nil)

func NewAdapter() *Adapter {
	return &Adapter{}
}

func (a *Adapter) SetLogger(logger *slog.Logger) {
	a.logger = logger
}

func (a *Adapter) Close() {
	a.clients.Range(func(key, val any) bool {
		_ = val.(*genai.Client).Close()

		// Continue iterating over the map.
		return true
	})
}

func (a *Adapter) ChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	contents := buildContent(req.Messages)
	model, err := a.loadOrStoreModel(ctx, req, isMultiModal(contents))
	if err != nil {
		return nil, err
	}

	contents, tail := pop(contents)

	// Chat messages must have roles alternating between 'user' and 'model'.
	sc := model.StartChat()
	sc.History = contents
	if a.logger != nil {
		a.logger.Info("sendMessage",
			slog.Any("contents", contents),
			slog.Any("tail", tail),
		)
	}

	// The send message must be from role `user`.
	resp, err := sc.SendMessage(ctx, tail.Parts...)
	if err != nil {
		return nil, err
	}

	return toOpenaiResponse(resp)
}

func (a *Adapter) ChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (chan openai.ChatCompletionStreamResponse, error) {
	contents := buildContent(req.Messages)
	model, err := a.loadOrStoreModel(ctx, req, isMultiModal(contents))
	if err != nil {
		return nil, err
	}

	contents, tail := pop(contents)

	// Chat messages must have roles alternating between 'user' and 'model'.
	sc := model.StartChat()
	sc.History = contents
	if a.logger != nil {
		a.logger.Info("sendMessage",
			slog.Any("contents", contents),
			slog.Any("tail", tail),
		)
	}

	ch := make(chan openai.ChatCompletionStreamResponse)
	go func() {
		iter := sc.SendMessageStream(ctx, tail.Parts...)

		for {
			res, err := iter.Next()
			if err == iterator.Done {
				close(ch)
				break
			}

			ch <- openai.ChatCompletionStreamResponse{
				ID:      "cmpl-" + uuid.New().String(),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   req.Model,
				Choices: toOpenaiStreamChoices(res.Candidates),
			}
		}
	}()

	return ch, nil
}

func (a *Adapter) createClient(ctx context.Context) (*genai.Client, error) {
	apiKey := ctx.Value(apiKeyContextKey).(string)
	openaiClient, ok := a.clients.Load(apiKey)
	if !ok {
		g, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
		if err != nil {
			return nil, err
		}

		c, loaded := a.clients.LoadOrStore(apiKey, g)
		if loaded {
			openaiClient = c
		} else {
			openaiClient = g
		}
	}

	return openaiClient.(*genai.Client), nil
}

func (a *Adapter) loadOrStoreModel(ctx context.Context, req openai.ChatCompletionRequest, isMultiModal bool) (*genai.GenerativeModel, error) {
	openaiClient, err := a.createClient(ctx)
	if err != nil {
		return nil, err
	}

	var model *genai.GenerativeModel
	if isMultiModal {
		model = openaiClient.GenerativeModel("gemini-pro-vision")
	} else {
		model = openaiClient.GenerativeModel("gemini-pro")
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

	if a.logger != nil {
		a.logger.Info("parameters",
			slog.Int("candidate_count", int(candidateCount)),
			slog.Int("max_output_tokens", int(maxOutputTokens)),
			slog.String("stop_sequences", strings.Join(stopSequences, " ")),
			slog.Float64("temperature", float64(temperature)),
			slog.Float64("top_p", float64(topP)),
			slog.Bool("isMultiModal", isMultiModal),
		)
	}

	return model, nil
}

func pop[T any](vs []T) ([]T, T) {
	if len(vs) == 0 {
		panic("pop from empty slice")
	}

	return vs[:len(vs)-1], vs[len(vs)-1]
}

func decodeBase64Image(b64 string) (mimeType string, blob []byte, err error) {
	lhs, rhs, ok := strings.Cut(b64, ";")
	if !ok {
		err = errors.New("invalid image format")
		return
	}

	mimeType = strings.ReplaceAll(lhs, "data:", "")
	b64Image := strings.ReplaceAll(rhs, "base64,", "")

	blob, err = base64.StdEncoding.DecodeString(b64Image)
	return
}
