package main

import (
	"DualPath/database"
	"DualPath/handlers"
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
	// Load .env file
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: .env file not found — using OS environment")
	}

	viper.AutomaticEnv()
	viper.SetDefault("PORT", "8080")
	viper.SetDefault("GIN_MODE", "debug")

	port := viper.GetString("PORT")

	if viper.GetString("GIN_MODE") == "release" {
		gin.SetMode(gin.ReleaseMode)
	}

	// Initialise Supabase PostgreSQL (pgvector handled inside)
	log.Println("Initialising Supabase connection…")
	if err := database.InitializePostgreSQL(); err != nil {
		log.Fatalf("PostgreSQL init failed: %v", err)
	}

	handlers.InitializeHandler()

	// Build router
	router := gin.Default()
	router.Use(gin.Recovery())

	// Health check
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "healthy",
			"services": gin.H{
				"supabase": "connected",
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

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 90 * time.Second,
	}

	go func() {
		log.Printf("DualPath (Supabase Edition) listening on :%s", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutdown signal received…")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("Forced shutdown: %v", err)
	}
	log.Println("Server stopped cleanly")
}

func getConversation(c *gin.Context) {
	id := c.Param("id")
	db := database.GetDB()
	var conversation models.Conversation
	if err := db.Where("conversation_id = ?", id).First(&conversation).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "conversation not found"})
		return
	}
	c.JSON(http.StatusOK, conversation)
}

func listConversations(c *gin.Context) {
	db := database.GetDB()
	var conversations []models.Conversation
	if err := db.Order("created_at desc").Find(&conversations).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"conversations": conversations, "total": len(conversations)})
}
