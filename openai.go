package goai

import (
	"log"
	"strings"

	"github.com/google/generative-ai-go/genai"
	openai "github.com/sashabaranov/go-openai"
)

var toOpenaiFinishReason = map[genai.FinishReason]openai.FinishReason{
	genai.FinishReasonUnspecified: openai.FinishReasonNull,
	genai.FinishReasonStop:        openai.FinishReasonStop,
	genai.FinishReasonMaxTokens:   openai.FinishReasonLength,
	genai.FinishReasonSafety:      openai.FinishReasonContentFilter,
	genai.FinishReasonRecitation:  openai.FinishReasonContentFilter,
	genai.FinishReasonOther:       openai.FinishReasonNull,
}

const (
	openaiRoleSystem    = "system"
	openaiRoleAssistant = "assistant"
	openaiRoleUser      = "user"
)

var toGenaiRole = map[string]string{
	openaiRoleSystem:    genaiRoleUser,
	openaiRoleAssistant: genaiRoleModel,
	openaiRoleUser:      genaiRoleUser,
}

var toOpenaiRole = map[string]string{
	genaiRoleUser:  openaiRoleSystem,
	genaiRoleModel: openaiRoleAssistant,
}

// mergeOpenaiMessages merge the messages with the same role. This ensures that there
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
func mergeOpenaiMessages(msgs []openai.ChatCompletionMessage) []openai.ChatCompletionMessage {
	var prevRole string
	var res []openai.ChatCompletionMessage

	for _, curr := range msgs {
		role, ok := toGenaiRole[curr.Role]
		if !ok {
			log.Fatalf("unknown openai role: %q", curr.Role)
		}

		if role == prevRole {
			// Merge the content if the roles are similar.
			prev := res[len(res)-1]

			pmc := prev.MultiContent
			cmc := curr.MultiContent

			pc := prev.Content
			cc := curr.Content

			lhs := len(pmc) > 0
			rhs := len(cmc) > 0

			switch {
			case lhs && rhs:
				pmc = append(pmc, cmc...)
			case lhs && !rhs:
				pmc = append(pmc, openai.ChatMessagePart{
					Type: openai.ChatMessagePartTypeText,
					Text: cc,
				})
			case !lhs && rhs:
				pmc = append([]openai.ChatMessagePart{{
					Type: openai.ChatMessagePartTypeText,
					Text: pc,
				}}, cmc...)
				pc = ""
			case !lhs && !rhs:
				pc = strings.Join([]string{pc, cc}, "\n")
			}

			prev.MultiContent = pmc
			prev.Content = pc
			res[len(res)-1] = prev
		} else {
			prevRole = role
			res = append(res, curr)
		}
	}

	return res
}

func toOpenaiResponse(resp *genai.GenerateContentResponse) (*openai.ChatCompletionResponse, error) {
	var res openai.ChatCompletionResponse
	res.Choices = make([]openai.ChatCompletionChoice, len(resp.Candidates))

	var tokens int
	for i, c := range resp.Candidates {
		tokens += int(c.TokenCount)

		res.Choices[i] = toOpenaiChoice(c)
	}

	res.Usage.CompletionTokens = tokens

	return &res, nil
}

func toOpenaiChoice(c *genai.Candidate) openai.ChatCompletionChoice {
	role := toOpenaiRole[c.Content.Role]
	index := int(c.Index)
	content := mergeText(c.Content.Parts)
	finishReason := toOpenaiFinishReason[c.FinishReason]

	return openai.ChatCompletionChoice{
		Index: index,
		Message: openai.ChatCompletionMessage{
			Role:    role,
			Content: content,
		},
		FinishReason: finishReason,
	}
}

func toOpenaiStreamChoices(candidates []*genai.Candidate) []openai.ChatCompletionStreamChoice {
	choices := make([]openai.ChatCompletionStreamChoice, len(candidates))
	for i, c := range candidates {
		choices[i] = toOpenaiStreamChoice(c)
	}

	return choices
}

func toOpenaiStreamChoice(c *genai.Candidate) openai.ChatCompletionStreamChoice {
	index := int(c.Index)
	content := mergeText(c.Content.Parts)
	role := toOpenaiRole[c.Content.Role]
	finishReason := toOpenaiFinishReason[c.FinishReason]

	return openai.ChatCompletionStreamChoice{
		Index: index,
		Delta: openai.ChatCompletionStreamChoiceDelta{
			Content: content,
			Role:    role,
		},
		FinishReason: finishReason,
		// TODO: Complete the rest of the fields.
		// ContentFilterResults : ContentFilterResults
	}
}
