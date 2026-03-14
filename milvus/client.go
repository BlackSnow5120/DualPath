package milvus

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/spf13/viper"
	"github.com/joho/godotenv"
)

var MilvusClient client.Client
const (
	CollectionName = "document_vectors"
	VectorDimension = 1536 // OpenAI embedding dimension
)

// InitializeMilvus connects to Milvus
func InitializeMilvus() error {
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: .env file not found, using environment variables")
	}

	addr := viper.GetString("MILVUS_ADDR")
	if addr == "" {
		addr = "localhost:19530"
	}

	var err error
	MilvusClient, err = client.NewGrpcClient(context.Background(), addr)
	if err != nil {
		return fmt.Errorf("failed to connect to Milvus: %v", err)
	}

	// Check if collection exists, create if not
	hasCollection, _ := MilvusClient.HasCollection(context.Background(), CollectionName)
	if !hasCollection {
		if err := CreateCollection(); err != nil {
			return fmt.Errorf("failed to create collection: %v", err)
		}
	}

	log.Println("Milvus connected successfully")
	return nil
}

// CreateCollection creates the vector collection schema
func CreateCollection() error {
	schema := &entity.Schema{
		CollectionName: CollectionName,
		Description:    "Document vectors for semantic search",
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				AutoID:    false,
				TypeParams: map[string]string{
					"max_length": "100",
				},
			},
			{
				Name:       "document_id",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{
					"max_length": "100",
				},
			},
			{
				Name:       "page_number",
				DataType:   entity.FieldTypeInt64,
			},
			{
				Name:       "chunk_index",
				DataType:   entity.FieldTypeInt64,
			},
			{
				Name:     "vector",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", VectorDimension),
				},
			},
			{
				Name:     "text_content",
				DataType: entity.FieldTypeVarChar,
				TypeParams: map[string]string{
					"max_length": "65535",
				},
			},
		},
	}

	if err := MilvusClient.CreateCollection(context.Background(), schema, 2); err != nil {
		return err
	}

	// Create index on vector field
	idx, err := entity.NewIndexHNSW(entity.COSINE, map[string]string{
		"M":              "16",
		"efConstruction": "128",
	})
	if err != nil {
		return err
	}

	if err := MilvusClient.CreateIndex(context.Background(), CollectionName, "vector", idx, false); err != nil {
		return err
	}

	log.Printf("Collection '%s' created successfully", CollectionName)
	return nil
}

// InsertVectors inserts document chunks into Milvus
func InsertVectors(documentID string, vectors [][]float32, pageNumbers []int64, texts []string) ([]string, error) {
	vectorIDs := make([]string, len(vectors))
	for i := range vectors {
		vectorIDs[i] = fmt.Sprintf("%s_chunk_%d_page_%d", documentID, i, pageNumbers[i])
	}

	// Prepare data
	idCol := entity.NewColumnVarChar("id", vectorIDs)
	documentIDCol := entity.NewColumnVarChar("document_id", make([]string, len(vectors)))
	pageNumCol := entity.NewColumnInt64("page_number", pageNumbers)
	chunkIndexCol := entity.NewColumnInt64("chunk_index", make([]int64, len(vectors)))
	vecCol := entity.NewColumnFloatVector("vector", VectorDimension, vectors)
	textCol := entity.NewColumnVarChar("text_content", texts)

	// Fill document_id column
	for i := range vectorIDs {
		documentIDCol.Append(documentID)
		chunkIndexCol.Append(int64(i))
	}

	// Insert into Milvus
	_, err := MilvusClient.Insert(
		context.Background(),
		CollectionName,
		"",
		idCol,
		documentIDCol,
		pageNumCol,
		chunkIndexCol,
		vecCol,
		textCol,
	)

	if err != nil {
		return nil, err
	}

	return vectorIDs, nil
}

// SearchVectors performs semantic search in Milvus
func SearchVectors(queryVector []float32, topK int) ([]entity.SearchResult, error) {
	// Load collection into memory
	err := MilvusClient.LoadCollection(context.Background(), CollectionName, false)
	if err != nil {
		return nil, err
	}

	// Create search vectors
	searchVectors := []entity.Vector{entity.FloatVector(queryVector)}

	// Execute search
	sp, _ := entity.NewIndexHNSWSearchParam(128)
	results, err := MilvusClient.Search(
		context.Background(),
		CollectionName,
		[]string{},
		entity.NewColumnFloatVector("vector", VectorDimension, searchVectors),
		"text_content",
		topK,
		sp,
	)

	if err != nil {
		return nil, err
	}

	return results[0], nil
}
