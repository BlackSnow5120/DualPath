package handlers

import (
	"DualPath/database"
	"DualPath/milvus"
	"DualPath/models"
	"DualPath/retrieval"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

type PromptRequest struct {
	Prompt         string `json:"prompt" binding:"required"`
	TopK           int    `json:"top_k"`
	Model          string `json:"model"`
	ConversationID string `json:"conversation_id,omitempty"`
}

type PromptResponse struct {
	Prompt         string                   `json:"prompt"`
	Response       string                   `json:"response"`
	Context        []retrieval.SearchResult `json:"context"`
	ConversationID string                   `json:"conversation_id"`
	Timestamp      time.Time                `json:"timestamp"`
	Sources        []SourceInfo             `json:"sources"`
}

type SourceInfo struct {
	DocumentID string `json:"document_id"`
	PageNumber int    `json:"page_number"`
}

var hybridRetriever *retrieval.HybridRetriever

// InitializeHandler sets up the handlers
func InitializeHandler() {
	// Initialize retriever
	hybridRetriever = &retrieval.HybridRetriever{
		EmbeddingFunc: func(text string) ([]float32, error) {
			// TODO: Replace with actual embedding API call
			// This is a dummy implementation
			return make([]float32, 1536), nil
		},
	}
}

// HandlePrompt handles the prompt endpoint
func HandlePrompt(c *gin.Context) {
	var req PromptRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set defaults
	if req.TopK == 0 {
		req.TopK = 5
	}

	// Generate conversation ID if not provided
	if req.ConversationID == "" {
		req.ConversationID = uuid.New().String()
	}

	// Step 1: Perform Hybrid Retrieval to get context
	context, err := hybridRetriever.HybridSearch(req.Prompt, req.TopK)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Failed to perform hybrid search: %v", err),
		})
		return
	}

	// Step 2: Construct context string for LLM
	contextString := buildContextString(context)

	// Step 3: Call LLM with prompt + context
	llmResponse, err := callLLM(req.Prompt, contextString, req.Model)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Failed to generate LLM response: %v", err),
		})
		return
	}

	// Step 4: Extract source information
	sources := extractSources(context)

	// Step 5: Context serialization for storage
	contextJSON, _ := json.Marshal(context)
	sourceDocumentsJSON, _ := json.Marshal(sources)

	// Step 6: Save conversation to database
	db := database.GetDB()
	storagePrompt := req.Prompt
	storageResponse := llmResponse

	conversation := models.Conversation{
		ConversationID:  req.ConversationID,
		Prompt:          storagePrompt,
		Response:        storageResponse,
		ContextUsed:     string(contextJSON),
		SourceDocuments: string(sourceDocumentsJSON),
		Metadata:        fmt.Sprintf(`{"model": "%s", "top_k": %d}`, req.Model, req.TopK),
	}

	if err := db.Create(&conversation).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Failed to save conversation: %v", err),
		})
		return
	}

	// Step 7: Store in Redis for quick retrieval
	if redisClient := database.GetRedis(); redisClient != nil {
		ctx := context.Background()
		key := fmt.Sprintf("conversation:%s:%d", req.ConversationID, time.Now().Unix())
		if err := redisClient.Set(ctx, key, string(contextJSON), 24*time.Hour).Err(); err != nil {
			// Log error but don't fail the request
			fmt.Printf("Failed to cache context in Redis: %v\n", err)
		}
	}

	// Step 8: Store prompt + response in Vector DB for future retrieval
	// Store the prompt as a searchable vector
	if err := storeConversationInVectorDB(req.ConversationID, req.Prompt, llmResponse, contextString); err != nil {
		// Log error but don't fail the request
		fmt.Printf("Failed to store conversation in Vector DB: %v\n", err)
	}

	// Step 9: Return response
	response := PromptResponse{
		Prompt:         req.Prompt,
		Response:       llmResponse,
		Context:        context,
		ConversationID: req.ConversationID,
		Timestamp:      time.Now(),
		Sources:        sources,
	}

	c.JSON(http.StatusOK, response)
}

// HandleUploadDocument handles document upload and processing
func HandleUploadDocument(c *gin.Context) {
	documentID := c.PostForm("document_id")
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Save file
	filePath := fmt.Sprintf("/tmp/%s_%s", documentID, file.Filename)
	if err := c.SaveUploadedFile(file, filePath); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Process document (simplified - in production would parse PDF/excel etc.)
	// For now, we'll create dummy text pages
	textPages := []string{
		"This is page 1 content. It contains important information about the warranty policy.",
		"This is page 2 content. It discusses payment terms and conditions.",
		"This is page 3 content. It outlines the return and refund policy.",
	}

	// Process document with parallel workers
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

// HandleSearch allows searching for conversations
func HandleSearch(c *gin.Context) {
	query := c.Query("q")
	topK := 5

	if c.Query("top_k") != "" {
		fmt.Sscanf(c.Query("top_k"), "%d", &topK)
	}

	// Use hybrid retriever to search
	results, err := hybridRetriever.HybridSearch(query, topK)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Search failed: %v", err),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"query":   query,
		"results": results,
	})
}

// Helper functions
func buildContextString(context []retrieval.SearchResult) string {
	var sb strings.Builder
	sb.WriteString("Context from relevant documents:\n\n")

	for i, result := range context {
		sb.WriteString(fmt.Sprintf("[Source %d - Document: %s, Page: %d]\n", i+1, result.DocumentID, result.PageNumber))
		sb.WriteString(fmt.Sprintf("Summary: %s\n", result.Summary))
		if result.TextContent != "" {
			sb.WriteString(fmt.Sprintf("Content: %s\n", result.TextContent))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

func callLLM(prompt, context, model string) (string, error) {
	// TODO: Replace with actual LLM API call (OpenAI, Anthropic, etc.)
	// This is a dummy implementation

	fullPrompt := fmt.Sprintf("Context:\n%s\n\nQuestion: %s\n\nAnswer:", context, prompt)

	// Simulate LLM response
	response := fmt.Sprintf("Based on the context provided, here's the answer to '%s'. The relevant information can be found in the documents with the summaries provided.", prompt)

	return response, nil
}

func extractSources(context []retrieval.SearchResult) []SourceInfo {
	sourcesMap := make(map[string]SourceInfo)

	for _, result := range context {
		key := fmt.Sprintf("%s:%d", result.DocumentID, result.PageNumber)
		if _, exists := sourcesMap[key]; !exists {
			sourcesMap[key] = SourceInfo{
				DocumentID: result.DocumentID,
				PageNumber: result.PageNumber,
			}
		}
	}

	sources := make([]SourceInfo, 0, len(sourcesMap))
	for _, source := range sourcesMap {
		sources = append(sources, source)
	}

	return sources
}

func storeConversationInVectorDB(conversationID, prompt, response, contextString string) error {
	// Generate embedding for the prompt
	// TODO: Replace with actual embedding API
	promptVector := make([]float32, 1536)
	responseVector := make([]float32, 1536)

	// Store in Milvus
	// We store the prompt as searchable for future queries
	text := fmt.Sprintf("Q: %s\nA: %s", prompt, response)

	documentID := fmt.Sprintf("conv_%s", conversationID)
	pageNumbers := []int64{0}

	_, err := milvus.InsertVectors(documentID, [][]float32{promptVector}, pageNumbers, []string{text})
	if err != nil {
		return err
	}

	return nil
}
