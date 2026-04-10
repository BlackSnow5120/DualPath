package handlers

import (
	"DualPath/database"
	"DualPath/milvus"
	"DualPath/models"
	"DualPath/retrieval"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/spf13/viper"
)

// PromptRequest is the request body for POST /api/v1/prompt
type PromptRequest struct {
	Prompt         string `json:"prompt" binding:"required"`
	TopK           int    `json:"top_k"`
	Model          string `json:"model"`
	ConversationID string `json:"conversation_id,omitempty"`
}

// PromptResponse is the response body for POST /api/v1/prompt
type PromptResponse struct {
	Prompt         string                   `json:"prompt"`
	Response       string                   `json:"response"`
	Context        []retrieval.SearchResult `json:"context"`
	ConversationID string                   `json:"conversation_id"`
	Timestamp      time.Time                `json:"timestamp"`
	Sources        []SourceInfo             `json:"sources"`
}

// SourceInfo summarises a unique source document/page referenced in the response
type SourceInfo struct {
	DocumentID string `json:"document_id"`
	PageNumber int    `json:"page_number"`
}

var hybridRetriever *retrieval.HybridRetriever

// InitializeHandler wires up the embedding function into the retriever.
func InitializeHandler() {
	hybridRetriever = &retrieval.HybridRetriever{
		EmbeddingFunc: openAIEmbedding,
	}
}

// HandlePrompt runs the full RAG pipeline: retrieve → LLM → store → respond.
func HandlePrompt(c *gin.Context) {
	var req PromptRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.TopK == 0 {
		req.TopK = 5
	}
	if req.ConversationID == "" {
		req.ConversationID = uuid.New().String()
	}

	// Step 1: Hybrid retrieval
	// FIX: renamed from "context" to "searchResults" — the original shadowed the
	// built-in "context" package name, causing a compile error on context.Background().
	searchResults, err := hybridRetriever.HybridSearch(req.Prompt, req.TopK)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("hybrid search failed: %v", err),
		})
		return
	}

	// Step 2: Build context string for the LLM
	contextString := buildContextString(searchResults)

	// Step 3: Call LLM (OpenAI or Anthropic depending on model name)
	llmResponse, err := callLLM(req.Prompt, contextString, req.Model)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("LLM call failed: %v", err),
		})
		return
	}

	// Step 4: Extract unique source references
	sources := extractSources(searchResults)

	// Step 5: Serialise context for storage
	contextJSON, _ := json.Marshal(searchResults)
	sourceDocsJSON, _ := json.Marshal(sources)

	// Step 6: Persist conversation to PostgreSQL
	db := database.GetDB()
	conversation := models.Conversation{
		ConversationID:  req.ConversationID,
		Prompt:          req.Prompt,
		Response:        llmResponse,
		ContextUsed:     string(contextJSON),
		SourceDocuments: string(sourceDocsJSON),
		Metadata:        fmt.Sprintf(`{"model":%q,"top_k":%d}`, req.Model, req.TopK),
	}
	if err := db.Create(&conversation).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("failed to save conversation: %v", err),
		})
		return
	}

	// Step 7: Cache in Redis (best-effort, non-blocking)
	if redisClient := database.GetRedis(); redisClient != nil {
		// FIX: use bgCtx instead of "ctx" / "context" to avoid shadowing
		bgCtx := context.Background()
		key := fmt.Sprintf("conversation:%s:%d", req.ConversationID, time.Now().Unix())
		if err := redisClient.Set(bgCtx, key, string(contextJSON), 24*time.Hour).Err(); err != nil {
			log.Printf("[Redis] Cache write failed: %v", err)
		}
	}

	// Step 8: Store prompt+response embedding in Vector DB (best-effort)
	if err := storeConversationInVectorDB(req.ConversationID, req.Prompt, llmResponse); err != nil {
		log.Printf("[VectorDB] Conversation storage failed: %v", err)
	}

	// Step 9: Return response
	c.JSON(http.StatusOK, PromptResponse{
		Prompt:         req.Prompt,
		Response:       llmResponse,
		Context:        searchResults,
		ConversationID: req.ConversationID,
		Timestamp:      time.Now(),
		Sources:        sources,
	})
}

// HandleUploadDocument saves an uploaded file and processes it in parallel.
func HandleUploadDocument(c *gin.Context) {
	documentID := c.PostForm("document_id")
	if documentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "document_id is required"})
		return
	}

	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// FIX: use os.TempDir() for cross-platform compatibility (original hard-coded /tmp/)
	filePath := filepath.Join(os.TempDir(), fmt.Sprintf("%s_%s", documentID, filepath.Base(file.Filename)))
	if err := c.SaveUploadedFile(file, filePath); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// TODO: Replace with real PDF/Excel parser; these are placeholder pages
	textPages := []string{
		"This is page 1 content. It contains important information about the warranty policy.",
		"This is page 2 content. It discusses payment terms and conditions.",
		"This is page 3 content. It outlines the return and refund policy.",
	}

	if err := retrieval.ProcessDocumentParallel(documentID, filePath, textPages); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":         "Document processed successfully",
		"document_id":     documentID,
		"pages_processed": len(textPages),
	})
}

// HandleSearch performs a hybrid search and returns raw results.
func HandleSearch(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query parameter 'q' is required"})
		return
	}

	topK := 5
	if raw := c.Query("top_k"); raw != "" {
		fmt.Sscanf(raw, "%d", &topK)
	}

	results, err := hybridRetriever.HybridSearch(query, topK)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("search failed: %v", err)})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"query":   query,
		"results": results,
		"count":   len(results),
	})
}

// --- Helper functions ---

func buildContextString(results []retrieval.SearchResult) string {
	var sb strings.Builder
	sb.WriteString("Context from relevant documents:\n\n")
	for i, r := range results {
		sb.WriteString(fmt.Sprintf("[Source %d | doc: %s | page: %d | score: %.2f | via: %s]\n",
			i+1, r.DocumentID, r.PageNumber, r.Score, r.Source))
		if r.Summary != "" {
			sb.WriteString(fmt.Sprintf("Summary: %s\n", r.Summary))
		}
		if r.TextContent != "" {
			sb.WriteString(fmt.Sprintf("Content: %s\n", r.TextContent))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// callLLM dispatches to the appropriate API based on the model name prefix.
func callLLM(prompt, contextStr, model string) (string, error) {
	if model == "" {
		model = "gpt-4"
	}
	if strings.HasPrefix(strings.ToLower(model), "claude") {
		return callAnthropicLLM(prompt, contextStr, model)
	}
	return callOpenAILLM(prompt, contextStr, model)
}

// callOpenAILLM calls the OpenAI Chat Completions API.
func callOpenAILLM(prompt, contextStr, model string) (string, error) {
	apiKey := viper.GetString("OPENAI_API_KEY")
	if apiKey == "" {
		log.Println("[LLM] OPENAI_API_KEY not set — returning stub response")
		return fmt.Sprintf(
			"[Stub] Answer for: %q — set OPENAI_API_KEY to enable real LLM responses.", prompt,
		), nil
	}

	payload := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "system", "content": "You are a helpful assistant. Answer questions based solely on the provided document context."},
			{"role": "user", "content": fmt.Sprintf("Context:\n%s\n\nQuestion: %s", contextStr, prompt)},
		},
		"max_tokens":  1000,
		"temperature": 0.7,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: 60 * time.Second}).Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if result.Error != nil {
		return "", fmt.Errorf("OpenAI error: %s", result.Error.Message)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("OpenAI returned no choices")
	}
	return strings.TrimSpace(result.Choices[0].Message.Content), nil
}

// callAnthropicLLM calls the Anthropic Messages API.
func callAnthropicLLM(prompt, contextStr, model string) (string, error) {
	apiKey := viper.GetString("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Println("[LLM] ANTHROPIC_API_KEY not set — returning stub response")
		return fmt.Sprintf(
			"[Stub] Answer for: %q — set ANTHROPIC_API_KEY to enable real LLM responses.", prompt,
		), nil
	}

	payload := map[string]interface{}{
		"model":      model,
		"max_tokens": 1024,
		"system":     "You are a helpful assistant. Answer questions based solely on the provided document context.",
		"messages": []map[string]string{
			{"role": "user", "content": fmt.Sprintf("Context:\n%s\n\nQuestion: %s", contextStr, prompt)},
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: 60 * time.Second}).Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if result.Error != nil {
		return "", fmt.Errorf("Anthropic error: %s", result.Error.Message)
	}
	if len(result.Content) == 0 {
		return "", fmt.Errorf("Anthropic returned no content")
	}
	return strings.TrimSpace(result.Content[0].Text), nil
}

func extractSources(results []retrieval.SearchResult) []SourceInfo {
	seen := make(map[string]bool)
	var sources []SourceInfo
	for _, r := range results {
		key := fmt.Sprintf("%s:%d", r.DocumentID, r.PageNumber)
		if !seen[key] {
			seen[key] = true
			sources = append(sources, SourceInfo{DocumentID: r.DocumentID, PageNumber: r.PageNumber})
		}
	}
	return sources
}

// storeConversationInVectorDB embeds the prompt+response pair and stores it in
// Milvus so it can be retrieved as context in future queries.
//
// FIX: original declared but never used responseVector; also passed contextString
// which was an unused parameter.
func storeConversationInVectorDB(conversationID, prompt, response string) error {
	text := fmt.Sprintf("Q: %s\nA: %s", prompt, response)

	vec, err := hybridRetriever.EmbeddingFunc(text)
	if err != nil {
		return fmt.Errorf("embedding failed: %v", err)
	}

	_, err = milvus.InsertVectors(
		fmt.Sprintf("conv_%s", conversationID),
		[][]float32{vec},
		[]int64{0},
		[]string{text},
	)
	return err
}

// openAIEmbedding calls the OpenAI Embeddings API (text-embedding-ada-002).
// Falls back to a zero vector when OPENAI_API_KEY is unset, so the app remains
// functional in development without an API key.
func openAIEmbedding(text string) ([]float32, error) {
	apiKey := viper.GetString("OPENAI_API_KEY")
	if apiKey == "" {
		log.Println("[Embedding] OPENAI_API_KEY not set — using zero vector")
		return make([]float32, milvus.VectorDimension), nil
	}

	payload := map[string]interface{}{
		"model": "text-embedding-ada-002",
		"input": text,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if result.Error != nil {
		return nil, fmt.Errorf("OpenAI embedding error: %s", result.Error.Message)
	}
	if len(result.Data) == 0 || len(result.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("OpenAI returned empty embedding")
	}
	return result.Data[0].Embedding, nil
}
