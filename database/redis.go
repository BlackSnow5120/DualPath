package database

import (
	"context"
	"fmt"
	"log"

	"github.com/go-redis/redis/v8"
	"github.com/spf13/viper"
)

var RedisClient *redis.Client

// Ctx is the shared background context for Redis operations
var Ctx = context.Background()

// InitializeRedis connects to Redis and validates the connection
func InitializeRedis() error {
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

	// Validate connection
	if _, err := RedisClient.Ping(Ctx).Result(); err != nil {
		return fmt.Errorf("failed to connect to Redis at %s: %v", addr, err)
	}

	log.Printf("Redis connected: %s (db=%d)", addr, db)
	return nil
}

// GetRedis returns the global Redis client
func GetRedis() *redis.Client {
	return RedisClient
}
