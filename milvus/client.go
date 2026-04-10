package milvus

import (
	"context"
	"fmt"
	"log"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/spf13/viper"
)

var MilvusClient client.Client

const (
	CollectionName  = "document_vectors"
	VectorDimension = 1536 // OpenAI text-embedding-ada-002 dimension
)

// VectorSearchResult is a processed single result from a Milvus search.
// Using a local type avoids leaking the SDK's entity.SearchResult into higher layers.
type VectorSearchResult struct {
	VectorID    string
	DocumentID  string
	PageNumber  int64
	TextContent string
	Score       float32
}

// InitializeMilvus connects to Milvus and ensures the collection is ready.
// Note: godotenv.Load() is called once in main.go; viper.AutomaticEnv() makes
// env vars available here without a redundant Load call.
func InitializeMilvus() error {
	addr := viper.GetString("MILVUS_ADDR")
	if addr == "" {
		addr = "localhost:19530"
	}

	var err error
	MilvusClient, err = client.NewGrpcClient(context.Background(), addr)
	if err != nil {
		return fmt.Errorf("failed to connect to Milvus at %s: %v", addr, err)
	}

	// Create collection if it does not exist
	hasCollection, err := MilvusClient.HasCollection(context.Background(), CollectionName)
	if err != nil {
		return fmt.Errorf("failed to check Milvus collection: %v", err)
	}
	if !hasCollection {
		if err := CreateCollection(); err != nil {
			return fmt.Errorf("failed to create Milvus collection: %v", err)
		}
	}

	log.Printf("Milvus connected: %s (collection: %s)", addr, CollectionName)
	return nil
}

// CreateCollection creates the vector collection schema and HNSW index.
func CreateCollection() error {
	schema := &entity.Schema{
		CollectionName: CollectionName,
		Description:    "Document vectors for semantic search",
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				AutoID:     false,
				TypeParams: map[string]string{"max_length": "200"},
			},
			{
				Name:       "document_id",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "100"},
			},
			{
				Name:     "page_number",
				DataType: entity.FieldTypeInt64,
			},
			{
				Name:     "chunk_index",
				DataType: entity.FieldTypeInt64,
			},
			{
				Name:     "vector",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", VectorDimension),
				},
			},
			{
				Name:       "text_content",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
		},
	}

	if err := MilvusClient.CreateCollection(context.Background(), schema, 2); err != nil {
		return err
	}

	// FIX: entity.NewIndexHNSW takes (metricType, M int, efConstruction int),
	// NOT a map[string]string like the old code had.
	idx, err := entity.NewIndexHNSW(entity.COSINE, 16, 128)
	if err != nil {
		return fmt.Errorf("failed to create HNSW index config: %v", err)
	}

	if err := MilvusClient.CreateIndex(
		context.Background(), CollectionName, "vector", idx, false,
	); err != nil {
		return fmt.Errorf("failed to create vector index: %v", err)
	}

	log.Printf("Milvus collection '%s' created with HNSW index", CollectionName)
	return nil
}

// InsertVectors inserts document chunks into the Milvus collection.
// FIX: the original code created columns pre-filled with empty values and then
// appended more in a second loop, producing double-length columns that corrupted
// the insert. Now columns are built in a single pass.
func InsertVectors(
	documentID string,
	vectors [][]float32,
	pageNumbers []int64,
	texts []string,
) ([]string, error) {
	n := len(vectors)
	vectorIDs := make([]string, n)
	docIDs := make([]string, n)
	chunkIdxs := make([]int64, n)

	// Build all column data in one pass
	for i := range vectors {
		vectorIDs[i] = fmt.Sprintf("%s_chunk_%d_page_%d", documentID, i, pageNumbers[i])
		docIDs[i] = documentID
		chunkIdxs[i] = int64(i)
	}

	idCol := entity.NewColumnVarChar("id", vectorIDs)
	documentIDCol := entity.NewColumnVarChar("document_id", docIDs)
	pageNumCol := entity.NewColumnInt64("page_number", pageNumbers)
	chunkIndexCol := entity.NewColumnInt64("chunk_index", chunkIdxs)
	vecCol := entity.NewColumnFloatVector("vector", VectorDimension, vectors)
	textCol := entity.NewColumnVarChar("text_content", texts)

	if _, err := MilvusClient.Insert(
		context.Background(),
		CollectionName,
		"", // default partition
		idCol, documentIDCol, pageNumCol, chunkIndexCol, vecCol, textCol,
	); err != nil {
		return nil, err
	}

	return vectorIDs, nil
}

// SearchVectors performs a semantic similarity search in Milvus and returns
// processed results — no raw SDK types leak to callers.
//
// FIX: the original code passed entity.NewColumnFloatVector as the query vector,
// which is the wrong type. The SDK expects []entity.Vector. The metric type and
// vector field name were also missing from the call.
func SearchVectors(queryVector []float32, topK int) ([]VectorSearchResult, error) {
	ctx := context.Background()

	// Load collection into memory before searching
	if err := MilvusClient.LoadCollection(ctx, CollectionName, false); err != nil {
		return nil, fmt.Errorf("failed to load Milvus collection: %v", err)
	}

	// Search parameters for HNSW (ef = 128 — exploration factor at query time)
	sp, err := entity.NewIndexHNSWSearchParam(128)
	if err != nil {
		return nil, fmt.Errorf("failed to create HNSW search params: %v", err)
	}

	// Correct SDK call: []entity.Vector, explicit field name, metric type, topK
	results, err := MilvusClient.Search(
		ctx,
		CollectionName,
		[]string{},   // partition names (empty = all)
		"",           // filter expression
		[]string{"document_id", "page_number", "text_content"}, // output fields
		[]entity.Vector{entity.FloatVector(queryVector)},
		"vector",
		entity.COSINE,
		topK,
		sp,
	)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 || results[0].ResultCount == 0 {
		return nil, nil
	}

	sr := results[0] // single query vector → single result set

	// ID column
	idCol, ok := sr.IDs.(*entity.ColumnVarChar)
	if !ok {
		return nil, fmt.Errorf("unexpected ID column type: %T", sr.IDs)
	}
	ids := idCol.Data()

	// Extract output field columns from FieldData by name
	var docIDData []string
	var pageNumData []int64
	var textData []string

	for _, col := range sr.Fields {
		switch col.Name() {
		case "document_id":
			if c, ok := col.(*entity.ColumnVarChar); ok {
				docIDData = c.Data()
			}
		case "page_number":
			if c, ok := col.(*entity.ColumnInt64); ok {
				pageNumData = c.Data()
			}
		case "text_content":
			if c, ok := col.(*entity.ColumnVarChar); ok {
				textData = c.Data()
			}
		}
	}

	processed := make([]VectorSearchResult, 0, sr.ResultCount)
	for i := 0; i < sr.ResultCount && i < len(ids); i++ {
		res := VectorSearchResult{VectorID: ids[i]}
		if i < len(docIDData) {
			res.DocumentID = docIDData[i]
		}
		if i < len(pageNumData) {
			res.PageNumber = pageNumData[i]
		}
		if i < len(textData) {
			res.TextContent = textData[i]
		}
		if i < len(sr.Scores) {
			res.Score = sr.Scores[i]
		}
		processed = append(processed, res)
	}

	return processed, nil
}
