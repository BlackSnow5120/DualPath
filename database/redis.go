package database

import (
	"context"
	"fmt"
	"log"

	"github.com/go-redis/redis/v8"
	"github.com/spf13/viper"
	"github.com/joho/godotenv"
)

var RedisClient *redis.Client
var Ctx = context.Background()

// InitializeRedis connects to Redis
func InitializeRedis() error {
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: .env file not found, using environment variables")
	}

	addr := viper.GetString("REDIS_ADDR")
	if addr == "" {
		addr = "localhost:6379"
	}

	password := viper.GetString("REDIS_PASSWORD")
	db := viper.GetInt("REDIS_DB")

	RedisClient = redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})

	// Test connection
	_, err := RedisClient.Ping(Ctx).Result()
	if err != nil {
		return fmt.Errorf("failed to connect to Redis: %v", err)
	}

	log.Println("Redis connected successfully")
	return nil
}

// GetRedis returns the Redis client
func GetRedis() *redis.Client {
	return RedisClient
}
