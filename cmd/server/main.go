package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"strings"

	goai "github.com/alextanhongpin/go-gemini"
	"github.com/sashabaranov/go-openai"
)

type openaiClient interface {
	ChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
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
