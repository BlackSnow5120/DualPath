package retrieval

import (
	"context"
	"fmt"
	"DualPath/models"
	"DualPath/database"
	"DualPath/milvus"
	"log"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// HybridRetriever combines Page Index (deterministic) and Vector DB (probabilistic)
type HybridRetriever struct {
	EmbeddingFunc func(text string) ([]float32, error)
}

// SearchResult combines both PageIndex and Vector results
type SearchResult struct {
	DocumentID  string  `json:"document_id"`
	PageNumber  int     `json:"page_number"`
	Summary     string  `json:"summary"`
	Keywords    string  `json:"keywords"`
	TextContent string  `json:"text_content"`
	OffsetStart int64   `json:"offset_start"`
	OffsetEnd   int64   `json:"offset_end"`
	Score       float32 `json:"score"`
	Source      string  `json:"source"` // "page_index" or "vector_db"
}

// ProcessDocumentParallel processes a document with two parallel goroutines
func ProcessDocumentParallel(documentID, filePath string, textPages []string) error {
	var wg sync.WaitGroup
	wg.Add(2)

	// Error channels
	vectorErr := make(chan error, 1)
	indexErr := make(chan error, 1)

	// Goroutine 1: Vector Worker - Send chunks to embedding model and Milvus
	go func() {
		defer wg.Done()
		log.Printf("[Vector Worker] Starting vector processing for document: %s", documentID)

		// Simulate embedding and vector insertion
		// In production, call real embedding API here
		chunkSize := 512
		var vectors [][]float32
		var pageNumbers []int64
		var texts []string

		for pageNum, pageText := range textPages {
			chunks := chunkText(pageText, chunkSize)
			for _, chunk := range chunks {
				vectors = append(vectors, make([]float32, milvus.VectorDimension))
				pageNumbers = append(pageNumbers, int64(pageNum+1))
				texts = append(texts, chunk)
			}
		}

		// Insert into Milvus
		_, err := milvus.InsertVectors(documentID, vectors, pageNumbers, texts)
		if err != nil {
			vectorErr <- fmt.Errorf("vector processing failed: %v", err)
			return
		}

		log.Printf("[Vector Worker] Completed vector processing for document: %s", documentID)
		vectorErr <- nil
	}()

	// Goroutine 2: Index Worker - Extract structural metadata to PostgreSQL
	go func() {
		defer wg.Done()
		log.Printf("[Index Worker] Starting structured indexing for document: %s", documentID)

		db := database.GetDB()

		// Create document record
		document := models.Document{
			DocumentID:  documentID,
			FileName:    filePath[strings.LastIndex(filePath, "/")+1:],
			FilePath:    filePath,
			TotalPages:  len(textPages),
			IsProcessed: true,
		}

		if err := db.Create(&document).Error; err != nil {
			indexErr <- fmt.Errorf("failed to create document: %v", err)
			return
		}

		// Create page indexes
		for pageNum, pageText := range textPages {
			keywords := extractKeywords(pageText)
			summary := generateSummary(pageText)

			pageIndex := models.PageIndex{
				DocumentID:  documentID,
				PageNumber:  pageNum + 1,
				Keywords:    strings.Join(keywords, ", "),
				Summary:     summary,
				OffsetStart: int64(pageNum * 1000), // Simplified offset calculation
				OffsetEnd:   int64((pageNum + 1) * 1000),
				Metadata:    fmt.Sprintf(`{"headers": [], "tables": %d}`, countTables(pageText)),
			}

			if err := db.Create(&pageIndex).Error; err != nil {
				indexErr <- fmt.Errorf("failed to create page index: %v", err)
				return
			}
		}

		log.Printf("[Index Worker] Completed structured indexing for document: %s", documentID)
		indexErr <- nil
	}()

	// Wait for both workers
	wg.Wait()

	// Check for errors
	if err := <-vectorErr; err != nil {
		return err
	}
	if err := <-indexErr; err != nil {
		return err
	}

	return nil
}

// HybridSearch performs hybrid retrieval combining deterministic and probabilistic search
func (hr *HybridRetriever) HybridSearch(query string, topK int) ([]SearchResult, error) {
	var results []SearchResult
	db := database.GetDB()

	// Step 1: Direct Lookup - Check if query mentions specific page or section
	if pageIndexResults, found := hr.directPageLookup(query); found {
		results = append(results, pageIndexResults...)
		log.Printf("[Hybrid Search] Found direct page lookup results: %d", len(pageIndexResults))
	}

	// Step 2: Semantic Search - Query Milvus for broad queries
	if hr.isSemanticQuery(query) {
		queryVector, err := hr.EmbeddingFunc(query)
		if err != nil {
			log.Printf("[Hybrid Search] Failed to generate embedding: %v", err)
		} else {
			vectorResults, err := milvus.SearchVectors(queryVector, topK)
			if err != nil {
				log.Printf("[Hybrid Search] Vector search failed: %v", err)
			} else {
				// Re-anchor vector results to their original locations
				anchoredResults := hr.anchorVectorResults(vectorResults)
				results = append(results, anchoredResults...)
				log.Printf("[Hybrid Search] Found vector search results: %d", len(anchoredResults))
			}
		}
	}

	// Step 3: Context Merging - Combine and deduplicate results
	results = hr.mergeResults(results, topK)

	return results, nil
}

// directPageLookup checks if query mentions a specific page
func (hr *HybridRetriever) directPageLookup(query string) ([]SearchResult, bool) {
	db := database.GetDB()

	// Extract page numbers from query
	pagePattern := regexp.MustCompile(`page\s*(\d+)`)
	matches := pagePattern.FindAllStringSubmatch(strings.ToLower(query), -1)

	if len(matches) == 0 {
		return nil, false
	}

	var results []SearchResult
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}

		pageNum, err := strconv.Atoi(match[1])
		if err != nil {
			continue
		}

		var pageIndexes []models.PageIndex
		db.Where("page_number = ?", pageNum).Find(&pageIndexes)

		for _, idx := range pageIndexes {
			results = append(results, SearchResult{
				DocumentID:  idx.DocumentID,
				PageNumber:  idx.PageNumber,
				Summary:     idx.Summary,
				Keywords:    idx.Keywords,
				OffsetStart: idx.OffsetStart,
				OffsetEnd:   idx.OffsetEnd,
				Score:       1.0, // Direct match gets highest score
				Source:      "page_index",
			})
		}
	}

	return results, len(results) > 0
}

// isSemanticQuery checks if query is broad enough for semantic search
func (hr *HybridRetriever) isSemanticQuery(query string) bool {
	// Check if query contains keywords that indicate semantic search
	semanticIndicators := []string{
		"explain", "describe", "what is", "how does", "why", "tell me",
		"summary", "overview", "information about",
	}

	lowerQuery := strings.ToLower(query)
	for _, indicator := range semanticIndicators {
		if strings.Contains(lowerQuery, indicator) {
			return true
		}
	}

	return false
}

// anchorVectorResults re-anchors vector results to Page Index
func (hr *HybridRetriever) anchorVectorResults(vectorResults []entity.SearchResult) []SearchResult {
	db := database.GetDB()
	var results []SearchResult

	for _, vResult := range vectorResults {
		documentID := vResult.ID.(string) // This should be the vector ID, need to parse

		// Parse document ID from vector ID (format: docID_chunk_N_page_N)
		docID := extractDocumentIDFromVectorID(vResult.ID.(string))
		pageNumber := extractPageNumberFromVectorID(vResult.ID.(string))

		// Get PageIndex entry for this document and page
		var pageIndex models.PageIndex
		if err := db.Where("document_id = ? AND page_number = ?", docID, pageNumber).First(&pageIndex).Error; err != nil {
			log.Printf("[Anchor Result] PageIndex not found for doc=%s page=%d: %v", docID, pageNumber, err)
			continue
		}

		results = append(results, SearchResult{
			DocumentID:  docID,
			PageNumber:  pageIndex.PageNumber,
			Summary:     pageIndex.Summary,
			Keywords:    pageIndex.Keywords,
			TextContent: vResult.Fields.GetColumn("text_content").(*entity.ColumnVarChar).Data()[0].(string),
			OffsetStart: pageIndex.OffsetStart,
			OffsetEnd:   pageIndex.OffsetEnd,
			Score:       vResult.Score,
			Source:      "vector_db",
		})
	}

	return results
}

// mergeResults combines and deduplicates results
func (hr *HybridRetriever) mergeResults(results []SearchResult, topK int) []SearchResult {
	// Sort by score (highest first)
	sortByScore(results)

	// Deduplicate by document_id + page_number
	seen := make(map[string]bool)
	var merged []SearchResult

	for _, result := range results {
		key := fmt.Sprintf("%s:%d", result.DocumentID, result.PageNumber)
		if !seen[key] {
			seen[key] = true
			merged = append(merged, result)
		}

		if len(merged) >= topK {
			break
		}
	}

	return merged
}

// Helper functions
func chunkText(text string, chunkSize int) []string {
	var chunks []string
	words := strings.Fields(text)

	for i := 0; i < len(words); i += chunkSize {
		end := i + chunkSize
		if end > len(words) {
			end = len(words)
		}
		chunks = append(chunks, strings.Join(words[i:end], " "))
	}

	return chunks
}

func extractKeywords(text string) []string {
	// Simplified keyword extraction
 words := strings.Fields(strings.ToLower(text))
 keywords := make(map[string]bool)

	stopWords := map[string]bool{
		"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
		"with", "by", "from", "of", "is", "are", "was", "were", "be", "been",
		"have", "has", "had", "do", "does", "did", "will", "would", "could",
		"should", "may", "might", "must", "this", "that", "these", "those",
	}

	for _, word := range words {
		if len(word) > 3 && !stopWords[word] {
			keywords[word] = true
		}
	}

	var result []string
	for word := range keywords {
		result = append(result, word)
	}

	return result
}

func generateSummary(text string) string {
	// Simplified summary generation
	words := strings.Split(text, " ")
	maxLen := 50
	if len(words) > maxLen {
		return strings.Join(words[:maxLen], " ") + "..."
	}
	return text
}

func countTables(text string) int {
	// Simplified table detection
	return strings.Count(text, "|") / 2
}

func sortByScore(results []SearchResult) {
	// Sort by score descending
	for i := 0; i < len(results)-1; i++ {
		for j := 0; j < len(results)-i-1; j++ {
			if results[j].Score < results[j+1].Score {
				results[j], results[j+1] = results[j+1], results[j]
			}
		}
	}
}

func extractDocumentIDFromVectorID(vectorID string) string {
	parts := strings.Split(vectorID, "_chunk_")
	if len(parts) > 0 {
		return parts[0]
	}
	return ""
}

func extractPageNumberFromVectorID(vectorID string) int {
	parts := strings.Split(vectorID, "_page_")
	if len(parts) > 1 {
		num, err := strconv.Atoi(strings.Split(parts[1], "_")[0])
		if err == nil {
			return num
		}
	}
	return 0
}
