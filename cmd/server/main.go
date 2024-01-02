package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"

	goai "github.com/alextanhongpin/go-gemini"
	"github.com/sashabaranov/go-openai"
)

type openaiClient interface {
	ChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
	ChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (chan openai.ChatCompletionStreamResponse, error)
}

var logger *slog.Logger

func init() {
	logger = slog.New(slog.NewJSONHandler(os.Stdout, nil))
}

func main() {
	a := goai.NewAdapter()
	a.SetLogger(logger)
	h := new(openaiHandler)
	h.adapter = a

	mux := http.NewServeMux()
	mux.HandleFunc("/chat/completions", h.ChatCompletion)
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

type openaiHandler struct {
	adapter openaiClient
}

func (h openaiHandler) ChatCompletion(w http.ResponseWriter, r *http.Request) {
	apiKey := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")

	ctx := r.Context()
	ctx = goai.AuthContext(ctx, apiKey)

	var req openai.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if req.Stream {
		h.streamResponse(ctx, w, req)
		return
	}

	res, err := h.adapter.ChatCompletion(ctx, req)
	if err != nil {
		logger.Error("chat completion failed",
			slog.String("error", err.Error()),
			slog.Any("request", req),
		)

		http.Error(w, err.Error(), http.StatusUnprocessableEntity)
		return
	}

	logger.Info("request", slog.Any("req", req), slog.Any("res", res))
	if err := json.NewEncoder(w).Encode(res); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

func (h openaiHandler) streamResponse(ctx context.Context, w http.ResponseWriter, req openai.ChatCompletionRequest) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Content-Type", "text/event-stream")

	ch, err := h.adapter.ChatCompletionStream(ctx, req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusPreconditionFailed)
		return
	}

	for res := range ch {
		b, err := json.Marshal(res)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		fmt.Fprintf(w, "data: %s \n\n", b)
		w.(http.Flusher).Flush()
	}

	fmt.Fprint(w, "data: [DONE] \n\n")
	w.(http.Flusher).Flush()
}
