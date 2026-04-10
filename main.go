package main

import (
	"DualPath/database"
	"DualPath/handlers"
	"DualPath/milvus"
	"DualPath/models"
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"github.com/spf13/viper"
)

func main() {
	// Load .env file once (sub-packages do NOT call godotenv.Load again)
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: .env file not found — falling back to OS environment variables")
	}

	// FIX: AutomaticEnv makes viper read values from OS environment variables.
	// Without this, viper.GetString() would always return "" (empty), so all
	// database/API credentials would be silently ignored.
	viper.AutomaticEnv()
	viper.SetDefault("PORT", "8080")
	viper.SetDefault("GIN_MODE", "debug")

	port := viper.GetString("PORT")
	if port == "" {
		port = "8080"
	}

	// Configure Gin mode from environment (set GIN_MODE=release in production)
	if viper.GetString("GIN_MODE") == "release" {
		gin.SetMode(gin.ReleaseMode)
	}

	// Initialise data stores
	log.Println("Initialising databases…")

	if err := database.InitializePostgreSQL(); err != nil {
		log.Fatalf("PostgreSQL init failed: %v", err)
	}
	if err := database.InitializeRedis(); err != nil {
		log.Fatalf("Redis init failed: %v", err)
	}
	if err := milvus.InitializeMilvus(); err != nil {
		log.Fatalf("Milvus init failed: %v", err)
	}

	handlers.InitializeHandler()

	// Build router
	router := gin.Default()
	router.Use(gin.Recovery())
	router.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Content-Type", "application/json")
		c.Next()
	})

	// Health check — reports connection status of all three stores
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "healthy",
			"services": gin.H{
				"postgresql": "connected",
				"redis":      "connected",
				"milvus":     "connected",
			},
		})
	})

	// API v1
	v1 := router.Group("/api/v1")
	{
		v1.POST("/documents/upload", handlers.HandleUploadDocument)
		v1.POST("/prompt", handlers.HandlePrompt)
		v1.GET("/search", handlers.HandleSearch)
		v1.GET("/conversations/:id", getConversation)
		v1.GET("/conversations", listConversations)
	}

	// HTTP server configured separately to enable graceful shutdown
	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 90 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in background goroutine
	go func() {
		log.Printf("DualPath API listening on :%s", port)
		log.Println("  GET  /health")
		log.Println("  POST /api/v1/documents/upload")
		log.Println("  POST /api/v1/prompt")
		log.Println("  GET  /api/v1/search?q=<query>")
		log.Println("  GET  /api/v1/conversations/:id")
		log.Println("  GET  /api/v1/conversations")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Graceful shutdown on SIGINT / SIGTERM
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutdown signal received — draining connections…")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("Forced shutdown: %v", err)
	}
	log.Println("Server stopped cleanly")
}

// getConversation retrieves a single conversation by ID.
func getConversation(c *gin.Context) {
	id := c.Param("id")
	db := database.GetDB()

	var conversation models.Conversation
	if err := db.Where("conversation_id = ?", id).First(&conversation).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "conversation not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"conversation_id":  conversation.ConversationID,
		"prompt":           conversation.Prompt,
		"response":         conversation.Response,
		"context_used":     conversation.ContextUsed,
		"source_documents": conversation.SourceDocuments,
		"metadata":         conversation.Metadata,
		"created_at":       conversation.CreatedAt,
	})
}

// listConversations returns all conversations with a short prompt preview.
func listConversations(c *gin.Context) {
	db := database.GetDB()

	var conversations []models.Conversation
	if err := db.Order("created_at desc").Find(&conversations).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	type ConversationSummary struct {
		ConversationID string    `json:"conversation_id"`
		CreatedAt      time.Time `json:"created_at"`
		PromptPreview  string    `json:"prompt_preview"`
	}

	summaries := make([]ConversationSummary, len(conversations))
	for i, conv := range conversations {
		preview := conv.Prompt
		if len(preview) > 100 {
			preview = preview[:100] + "…"
		}
		summaries[i] = ConversationSummary{
			ConversationID: conv.ConversationID,
			CreatedAt:      conv.CreatedAt,
			PromptPreview:  preview,
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"conversations": summaries,
		"total":         len(conversations),
	})
}
