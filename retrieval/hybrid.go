package retrieval

import (
	"DualPath/database"
	"DualPath/models"
	"fmt"
	"log"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/pgvector/pgvector-go"
)

// HybridRetriever combines Page Index (deterministic) and Vector DB (probabilistic).
type HybridRetriever struct {
	EmbeddingFunc func(text string) ([]float32, error)
}

// SearchResult is the unified result type returned from any retrieval path.
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

// ProcessDocumentParallel processes a document and stores it in Supabase.
// It uses parallel workers for embedding generation and structural extraction.
func ProcessDocumentParallel(documentID, filePath string, textPages []string, embedFunc func(string) ([]float32, error)) error {
	var wg sync.WaitGroup
	wg.Add(1) // Just one worker for document metadata + page processing

	indexErr := make(chan error, 1)

	// Goroutine: Process document metadata and each page (with embeddings)
	go func() {
		defer wg.Done()
		log.Printf("[Worker] Processing document: %s", documentID)

		db := database.GetDB()

		// 1. Create document record
		document := models.Document{
			DocumentID:  documentID,
			FileName:    filePath[strings.LastIndex(filePath, "/")+1:],
			FilePath:    filePath,
			TotalPages:  len(textPages),
			IsProcessed: true,
		}
		if err := db.Create(&document).Error; err != nil {
			indexErr <- fmt.Errorf("failed to create document record: %v", err)
			return
		}

		// 2. Process pages and generate embeddings
		// We can use a small worker pool here if textPages is large
		for pageNum, pageText := range textPages {
			keywords := extractKeywords(pageText)
			summary := generateSummary(pageText)

			// Generate embedding for the page summary or first chunk
			// In this Supabase version, we store one vector per page for simplicity.
			vec, err := embedFunc(pageText)
			if err != nil {
				log.Printf("[Worker] Embedding failed for page %d in doc %s: %v", pageNum+1, documentID, err)
				vec = make([]float32, 1536) // Fallback
			}

			pageIndex := models.PageIndex{
				DocumentID:  documentID,
				PageNumber:  pageNum + 1,
				Keywords:    strings.Join(keywords, ", "),
				Summary:     summary,
				OffsetStart: int64(pageNum * 1000),
				OffsetEnd:   int64((pageNum + 1) * 1000),
				Embedding:   pgvector.NewVector(vec),
				Metadata:    fmt.Sprintf(`{"headers":[],"tables":%d}`, countTables(pageText)),
			}

			if err := db.Create(&pageIndex).Error; err != nil {
				indexErr <- fmt.Errorf("failed to create page index (page %d): %v", pageNum+1, err)
				return
			}
		}

		log.Printf("[Worker] Done: %s", documentID)
		indexErr <- nil
	}()

	wg.Wait()
	return <-indexErr
}

// HybridSearch executes hybrid retrieval using Supabase pgvector:
//  1. Direct page-number lookup (deterministic, score=1.0)
//  2. Semantic vector search (via SQL operator <=>)
//  3. Merge, deduplicate, rank by score
func (hr *HybridRetriever) HybridSearch(query string, topK int) ([]SearchResult, error) {
	var results []SearchResult

	// Step 1: Deterministic lookup for explicit page references
	if pageResults, found := hr.directPageLookup(query); found {
		results = append(results, pageResults...)
	}

	// Step 2: Semantic search via pgvector
	if hr.isSemanticQuery(query) || len(results) == 0 {
		queryVector, err := hr.EmbeddingFunc(query)
		if err != nil {
			log.Printf("[HybridSearch] Embedding failed: %v", err)
		} else {
			vectorResults, err := hr.vectorSearch(queryVector, topK)
			if err != nil {
				log.Printf("[HybridSearch] Vector search failed: %v", err)
			} else {
				results = append(results, vectorResults...)
			}
		}
	}

	// Step 3: Merge and return top-K
	return hr.mergeResults(results, topK), nil
}

// vectorSearch performs a similarity search using pgvector SQL operators
func (hr *HybridRetriever) vectorSearch(queryVector []float32, topK int) ([]SearchResult, error) {
	db := database.GetDB()
	var pageIndexes []models.PageIndex

	// Use cosine distance (<=>) for similarity. 
	// The operator 1 - (<=>) gives us the similarity score.
	// We use Order with raw SQL because GORM v1 doesn't have a cleaner way for pgvector operators.
	vec := pgvector.NewVector(queryVector)
	
	err := db.Order(fmt.Sprintf("embedding <=> '%s'", vec.String())).Limit(topK).Find(&pageIndexes).Error
	if err != nil {
		return nil, err
	}

	var results []SearchResult
	for _, idx := range pageIndexes {
		// Calculate a rough score (1 - distance)
		// Note: To get precise distance we'd need a raw Select query, but this suffices for ranking.
		results = append(results, SearchResult{
			DocumentID:  idx.DocumentID,
			PageNumber:  idx.PageNumber,
			Summary:     idx.Summary,
			Keywords:    idx.Keywords,
			OffsetStart: idx.OffsetStart,
			OffsetEnd:   idx.OffsetEnd,
			Score:       0.9, // Placeholder since we just used OrderBy
			Source:      "vector_db",
		})
	}

	return results, nil
}

// directPageLookup checks whether the query explicitly names one or more pages
func (hr *HybridRetriever) directPageLookup(query string) ([]SearchResult, bool) {
	db := database.GetDB()
	pagePattern := regexp.MustCompile(`page\s*(\d+)`)
	matches := pagePattern.FindAllStringSubmatch(strings.ToLower(query), -1)

	if len(matches) == 0 {
		return nil, false
	}

	var results []SearchResult
	for _, match := range matches {
		if len(match) < 2 { continue }
		pageNum, err := strconv.Atoi(match[1])
		if err != nil { continue }

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
				Score:       1.0,
				Source:      "page_index",
			})
		}
	}

	return results, len(results) > 0
}

func (hr *HybridRetriever) isSemanticQuery(query string) bool {
	indicators := []string{
		"explain", "describe", "what is", "how does", "why", "tell me",
		"summary", "overview", "information about", "details about",
	}
	lowerQuery := strings.ToLower(query)
	for _, ind := range indicators {
		if strings.Contains(lowerQuery, ind) { return true }
	}
	return false
}

func (hr *HybridRetriever) mergeResults(results []SearchResult, topK int) []SearchResult {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	seen := make(map[string]bool)
	merged := make([]SearchResult, 0, topK)

	for _, r := range results {
		key := fmt.Sprintf("%s:%d", r.DocumentID, r.PageNumber)
		if !seen[key] {
			seen[key] = true
			merged = append(merged, r)
		}
		if len(merged) >= topK { break }
	}

	return merged
}

// Helpers...
func chunkText(text string, chunkSize int) []string {
	words := strings.Fields(text)
	var chunks []string
	for i := 0; i < len(words); i += chunkSize {
		end := i + chunkSize
		if end > len(words) { end = len(words) }
		chunks = append(chunks, strings.Join(words[i:end], " "))
	}
	return chunks
}

func extractKeywords(text string) []string {
	words := strings.Fields(strings.ToLower(text))
	keywords := make(map[string]bool)
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true,
		"but": true, "in": true, "on": true, "at": true, "to": true,
		"for": true, "with": true, "by": true, "from": true, "of": true,
		"is": true, "are": true, "was": true, "were": true, "be": true,
		"been": true, "have": true, "has": true, "had": true, "do": true,
		"does": true, "did": true, "will": true, "would": true, "could": true,
		"should": true, "may": true, "might": true, "must": true,
		"this": true, "that": true, "these": true, "those": true,
	}
	for _, word := range words {
		word = strings.Trim(word, ".,!?;:\"'()")
		if len(word) > 3 && !stopWords[word] { keywords[word] = true }
	}
	result := make([]string, 0, len(keywords))
	for w := range keywords { result = append(result, w) }
	return result
}

func generateSummary(text string) string {
	const maxWords = 50
	words := strings.Fields(text)
	if len(words) > maxWords { return strings.Join(words[:maxWords], " ") + "..." }
	return text
}

func countTables(text string) int {
	return strings.Count(text, "|") / 2
}
