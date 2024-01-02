package goai

import (
	"log"
	"strings"

	"github.com/google/generative-ai-go/genai"
	openai "github.com/sashabaranov/go-openai"
)

const systemPrompt = "I will ask you a question. Please answer it."

const (
	genaiRoleUser  = "user"
	genaiRoleModel = "model"
)

func buildContent(msgs []openai.ChatCompletionMessage) []*genai.Content {
	msgs = mergeOpenaiMessages(msgs)
	contents := toGenaiContents(msgs)
	return reorderContentByRole(contents)
}

func toGenaiContents(msgs []openai.ChatCompletionMessage) []*genai.Content {
	contents := make([]*genai.Content, len(msgs))

	for i, msg := range msgs {
		contents[i] = toGenaiContent(msg)
	}

	return contents
}

func toGenaiContent(msg openai.ChatCompletionMessage) *genai.Content {
	r := toGenaiRole[msg.Role]
	c := msg.Content
	mc := msg.MultiContent

	var parts []genai.Part
	if len(mc) == 0 {
		parts = append(parts, genai.Text(c))
	} else {
		parts = make([]genai.Part, len(mc))
		for j, content := range mc {
			parts[j] = toGenaiPart(content)
		}
	}

	return &genai.Content{
		Role:  r,
		Parts: parts,
	}
}

func toGenaiPart(mp openai.ChatMessagePart) genai.Part {
	switch mp.Type {
	case openai.ChatMessagePartTypeText:
		return genai.Text(mp.Text)

	case openai.ChatMessagePartTypeImageURL:
		return toGenaiImageData(mp.ImageURL.URL)

	default:
		panic("unhandled type")
	}
}

func toGenaiImageData(b64img string) genai.Part {
	mimeType, blob, err := decodeBase64Image(b64img)
	if err != nil {
		log.Fatalf("failed to decode base64 image: %v", err)
	}

	format := strings.TrimPrefix(mimeType, "image/")
	return genai.ImageData(format, blob)
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

func mergeText(parts []genai.Part) string {
	texts := make([]string, len(parts))
	for i, p := range parts {
		t, ok := p.(genai.Text)
		if !ok {
			panic("part is not text")
		}

		texts[i] = string(t)
	}

	return strings.Join(texts, "")
}

func reorderContentByRole(contents []*genai.Content) []*genai.Content {
	if contents[len(contents)-1].Role != genaiRoleUser {
		panic("last message must be from user")
	}

	if contents[0].Role == genaiRoleUser {
		return contents
	}

	return append([]*genai.Content{{
		Role:  genaiRoleUser,
		Parts: []genai.Part{genai.Text(systemPrompt)},
	}}, contents...)
}
