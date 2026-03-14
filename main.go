package main

import (
	"DualPath/database"
	"DualPath/handlers"
	"DualPath/milvus"
	"DualPath/models"
	"fmt"
	"log"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/spf13/viper"
	"github.com/joho/godotenv"
)

func main() {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: .env file not found, using environment variables")
	}

	// Set default values
	viper.SetDefault("PORT", "8080")
	port := viper.GetString("PORT")
	if port == "" {
		port = "8080"
	}

	// Initialize databases
	log.Println("Initializing databases...")

	// PostgreSQL
	if err := database.InitializePostgreSQL(); err != nil {
		log.Fatalf("Failed to initialize PostgreSQL: %v", err)
	}

	// Redis
	if err := database.InitializeRedis(); err != nil {
		log.Fatalf("Failed to initialize Redis: %v", err)
	}

	// Milvus
	if err := milvus.InitializeMilvus(); err != nil {
		log.Fatalf("Failed to initialize Milvus: %v", err)
	}

	// Initialize handlers
	handlers.InitializeHandler()

	// Setup Gin router
	router := gin.Default()

	// Middleware
	router.Use(gin.Recovery())
	router.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Content-Type", "application/json")
		c.Next()
	})

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status": "healthy",
			"services": gin.H{
				"postgresql": "connected",
				"redis":     "connected",
				"milvus":    "connected",
			},
		})
	})

	// API v1 routes
	v1 := router.Group("/api/v1")
	{
		// Document management
		v1.POST("/documents/upload", handlers.HandleUploadDocument)

		// Prompt and conversation endpoints
		v1.POST("/prompt", handlers.HandlePrompt)
		v1.GET("/search", handlers.HandleSearch)

		// Conversation history
		v1.GET("/conversations/:id", getConversation)
		v1.GET("/conversations", listConversations)
	}

	// Start server
	log.Printf("Server starting on port %s...", port)
	log.Println("Available endpoints:")
	log.Println("  GET  /health - Health check")
	log.Println("  POST /api/v1/documents/upload - Upload and process documents")
	log.Println("  POST /api/v1/prompt - Send prompt with context")
	log.Println("  GET  /api/v1/search - Search documents")
	log.Println("  GET  /api/v1/conversations/:id - Get specific conversation")
	log.Println("  GET  /api/v1/conversations - List all conversations")

	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

// getConversation retrieves a specific conversation
func getConversation(c *gin.Context) {
	id := c.Param("id")
	db := database.GetDB()

	var conversation models.Conversation
	if err := db.Where("conversation_id = ?", id).First(&conversation).Error; err != nil {
		c.JSON(404, gin.H{"error": "Conversation not found"})
		return
	}

	c.JSON(200, gin.H{
		"conversation_id": conversation.ConversationID,
		"prompt":          conversation.Prompt,
		"response":        conversation.Response,
		"context_used":    conversation.ContextUsed,
		"source_documents": conversation.SourceDocuments,
		"metadata":        conversation.Metadata,
		"created_at":      conversation.CreatedAt,
	})
}

// listConversations retrieves all conversations
func listConversations(c *gin.Context) {
	db := database.GetDB()

	var conversations []models.Conversation
	if err := db.Find(&conversations).Error; err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	// Build response
	type ConversationSummary struct {
		ConversationID string    `json:"conversation_id"`
		CreatedAt      time.Time `json:"created_at"`
		PromptPreview  string    `json:"prompt_preview"`
	}

	summaries := make([]ConversationSummary, len(conversations))
	for i, conv := range conversations {
		preview := conv.Prompt
		if len(preview) > 100 {
			preview = preview[:100] + "..."
		}
		summaries[i] = ConversationSummary{
			ConversationID: conv.ConversationID,
			CreatedAt:      conv.CreatedAt,
			PromptPreview:  preview,
		}
	}

	c.JSON(200, gin.H{
		"conversations": summaries,
		"total":         len(conversations),
	})
}
