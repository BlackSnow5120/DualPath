package database

import (
	"fmt"
	"DualPath/models"
	"log"

	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/postgres"
	"github.com/spf13/viper"
	"github.com/joho/godotenv"
)

var DB *gorm.DB

// InitializePostgreSQL connects to PostgreSQL and creates tables
func InitializePostgreSQL() error {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: .env file not found, using environment variables")
	}

	host := viper.GetString("DB_HOST")
	port := viper.GetString("DB_PORT")
	user := viper.GetString("DB_USER")
	password := viper.GetString("DB_PASSWORD")
	dbname := viper.GetString("DB_NAME")

	// Set defaults
	if host == "" {
		host = "localhost"
	}
	if port == "" {
		port = "5432"
	}
	if user == "" {
		user = "postgres"
	}
	if dbname == "" {
		dbname = "dualpath"
	}

	psqlInfo := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		host, port, user, password, dbname)

	var err error
	DB, err = gorm.Open("postgres", psqlInfo)
	if err != nil {
		return fmt.Errorf("failed to connect to database: %v", err)
	}

	// Auto-migrate models
	DB.AutoMigrate(&models.PageIndex{}, &models.Document{}, &models.Conversation{}, &models.VectorIndexLink{})

	log.Println("PostgreSQL connected and tables created successfully")
	return nil
}

// GetDB returns the database connection
func GetDB() *gorm.DB {
	return DB
}
