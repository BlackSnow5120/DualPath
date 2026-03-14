package models

import (
	"github.com/jinzhu/gorm"
	"time"
)

// PageIndex represents the deterministic index for document locations
type PageIndex struct {
	ID          uint      `gorm:"primary_key" json:"id"`
	DocumentID  string    `gorm:"index" json:"document_id" binding:"required"`
	PageNumber  int       `gorm:"index" json:"page_number" binding:"required"`
	Keywords    string    `gorm:"type:text" json:"keywords"`
	Summary     string    `gorm:"type:text" json:"summary"`
	OffsetStart int64     `json:"offset_start"`
	OffsetEnd   int64     `json:"offset_end"`
	Metadata    string    `gorm:"type:jsonb" json:"metadata"` // For storing additional data like headers, tables, etc.
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// Document represents the main document record
type Document struct {
	ID          uint      `gorm:"primary_key" json:"id"`
	DocumentID  string    `gorm:"unique_index" json:"document_id" binding:"required"`
	FileName    string    `json:"file_name"`
	ContentType string    `json:"content_type"`
	FilePath    string    `json:"file_path"`
	TotalPages  int       `json:"total_pages"`
	FileSize    int64     `json:"file_size"`
	IsProcessed bool      `gorm:"default:false" json:"is_processed"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	Pages       []PageIndex `gorm:"foreignKey:DocumentID" json:"pages,omitempty"`
}

// Conversation stores prompt and LLM response pairs for vector DB retrieval
type Conversation struct {
	ID          uint      `gorm:"primary_key" json:"id"`
	ConversationID string  `gorm:"index" json:"conversation_id" binding:"required"`
	Prompt      string    `gorm:"type:text" json:"prompt" binding:"required"`
	Response    string    `gorm:"type:text" json:"response" binding:"required"`
	ContextUsed string    `gorm:"type:text" json:"context_used"` // JSON string of context sources
	SourceDocuments string `gorm:"type:text" json:"source_documents"` // JSON array of document IDs used
	Metadata    string    `gorm:"type:jsonb" json:"metadata"` // Additional metadata
	CreatedAt   time.Time `json:"created_at"`
}

// VectorIndexLink links PageIndex entries to Milvus vector IDs
type VectorIndexLink struct {
	ID           uint   `gorm:"primary_key" json:"id"`
	PageIndexID  uint   `gorm:"index" json:"page_index_id" binding:"required"`
	VectorID     string `gorm:"index" json:"vector_id" binding:"required"` // Milvus vector ID
	ChunkIndex   int    `json:"chunk_index"`
	CreatedAt    time.Time `json:"created_at"`
}
