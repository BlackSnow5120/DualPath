package database

import (
	"DualPath/models"
	"fmt"
	"log"

	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/postgres"
	"github.com/spf13/viper"
)

var DB *gorm.DB

// InitializePostgreSQL connects to PostgreSQL and auto-migrates models
func InitializePostgreSQL() error {
	host := viper.GetString("DB_HOST")
	port := viper.GetString("DB_PORT")
	user := viper.GetString("DB_USER")
	password := viper.GetString("DB_PASSWORD")
	dbname := viper.GetString("DB_NAME")

	// Apply defaults for missing values
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

	dsn := fmt.Sprintf(
		"host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		host, port, user, password, dbname,
	)

	var err error
	DB, err = gorm.Open("postgres", dsn)
	if err != nil {
		return fmt.Errorf("failed to connect to PostgreSQL: %v", err)
	}

	// Enable connection pool settings
	DB.DB().SetMaxIdleConns(10)
	DB.DB().SetMaxOpenConns(100)

	// Auto-migrate all models
	DB.AutoMigrate(
		&models.PageIndex{},
		&models.Document{},
		&models.Conversation{},
		&models.VectorIndexLink{},
	)

	log.Printf("PostgreSQL connected: %s:%s/%s", host, port, dbname)
	return nil
}

// GetDB returns the global database connection
func GetDB() *gorm.DB {
	return DB
}
