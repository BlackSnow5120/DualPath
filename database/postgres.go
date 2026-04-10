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

// InitializePostgreSQL connects to Supabase PostgreSQL and auto-migrates models
func InitializePostgreSQL() error {
	// Support both individual params and full connection string
	dsn := viper.GetString("DATABASE_URL")
	if dsn == "" {
		host := viper.GetString("DB_HOST")
		port := viper.GetString("DB_PORT")
		user := viper.GetString("DB_USER")
		password := viper.GetString("DB_PASSWORD")
		dbname := viper.GetString("DB_NAME")

		if host == "" { host = "localhost" }
		if port == "" { port = "5432" }
		if user == "" { user = "postgres" }
		if dbname == "" { dbname = "dualpath" }

		dsn = fmt.Sprintf(
			"host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
			host, port, user, password, dbname,
		)
	}

	var err error
	DB, err = gorm.Open("postgres", dsn)
	if err != nil {
		return fmt.Errorf("failed to connect to PostgreSQL: %v", err)
	}

	// Enable connection pool settings
	DB.DB().SetMaxIdleConns(10)
	DB.DB().SetMaxOpenConns(100)

	// Ensure pgvector extension exists
	// We run this as raw SQL because GORM v1 doesn't have a native 'CreateExtension' helper
	if err := DB.Exec("CREATE EXTENSION IF NOT EXISTS vector").Error; err != nil {
		log.Printf("Warning: Failed to ensure 'vector' extension: %v (make sure it is enabled in Supabase)", err)
	}

	// Auto-migrate models
	DB.AutoMigrate(
		&models.PageIndex{},
		&models.Document{},
		&models.Conversation{},
	)

	log.Println("PostgreSQL (Supabase) connected and models migrated")
	return nil
}

// GetDB returns the global database connection
func GetDB() *gorm.DB {
	return DB
}
