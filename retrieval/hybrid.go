package retrieval

import (
	"DualPath/database"
	"DualPath/milvus"
	"DualPath/models"
	"fmt"
	"log"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
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

// ProcessDocumentParallel processes a document with two concurrent goroutines:
//   - Vector Worker  → embeds chunks and stores in Milvus
//   - Index Worker   → extracts structured metadata and stores in PostgreSQL
func ProcessDocumentParallel(documentID, filePath string, textPages []string, embedFunc func(string) ([]float32, error)) error {
	var wg sync.WaitGroup
	wg.Add(2)

	vectorErr := make(chan error, 1)
	indexErr := make(chan error, 1)

	// --- Goroutine 1: Vector Worker ---
	go func() {
		defer wg.Done()
		log.Printf("[Vector Worker] Processing document: %s", documentID)

		chunkSize := 512
		var vectors [][]float32
		var pageNumbers []int64
		var texts []string

		for pageNum, pageText := range textPages {
			for _, chunk := range chunkText(pageText, chunkSize) {
				vec, err := embedFunc(chunk)
				if err != nil {
					log.Printf("[Vector Worker] Embedding failed for chunk in doc %s: %v", documentID, err)
					vec = make([]float32, milvus.VectorDimension) // Fallback to zero vector
				}
				vectors = append(vectors, vec)
				pageNumbers = append(pageNumbers, int64(pageNum+1))
				texts = append(texts, chunk)
			}
		}

		if _, err := milvus.InsertVectors(documentID, vectors, pageNumbers, texts); err != nil {
			vectorErr <- fmt.Errorf("vector processing failed: %v", err)
			return
		}

		log.Printf("[Vector Worker] Done: %s", documentID)
		vectorErr <- nil
	}()

	// --- Goroutine 2: Index Worker ---
	go func() {
		defer wg.Done()
		log.Printf("[Index Worker] Processing document: %s", documentID)

		db := database.GetDB()

		// Record the document itself
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

		// Create one PageIndex row per page
		for pageNum, pageText := range textPages {
			keywords := extractKeywords(pageText)
			pageIndex := models.PageIndex{
				DocumentID:  documentID,
				PageNumber:  pageNum + 1,
				Keywords:    strings.Join(keywords, ", "),
				Summary:     generateSummary(pageText),
				OffsetStart: int64(pageNum * 1000),
				OffsetEnd:   int64((pageNum + 1) * 1000),
				Metadata:    fmt.Sprintf(`{"headers":[],"tables":%d}`, countTables(pageText)),
			}
			if err := db.Create(&pageIndex).Error; err != nil {
				indexErr <- fmt.Errorf("failed to create page index (page %d): %v", pageNum+1, err)
				return
			}
		}

		log.Printf("[Index Worker] Done: %s", documentID)
		indexErr <- nil
	}()

	wg.Wait()

	if err := <-vectorErr; err != nil {
		return err
	}
	return <-indexErr
}

// HybridSearch executes hybrid retrieval:
//  1. Direct page-number lookup (deterministic, score=1.0)
//  2. Semantic vector search (probabilistic, score=0–1)
//  3. Merge, deduplicate, rank by score
func (hr *HybridRetriever) HybridSearch(query string, topK int) ([]SearchResult, error) {
	var results []SearchResult

	// Step 1: Deterministic lookup for explicit page references
	if pageResults, found := hr.directPageLookup(query); found {
		results = append(results, pageResults...)
		log.Printf("[HybridSearch] Direct lookup: %d result(s)", len(pageResults))
	}

	// Step 2: Semantic search — run when query is broad OR no direct results found
	if hr.isSemanticQuery(query) || len(results) == 0 {
		queryVector, err := hr.EmbeddingFunc(query)
		if err != nil {
			log.Printf("[HybridSearch] Embedding failed: %v", err)
		} else {
			vectorResults, err := milvus.SearchVectors(queryVector, topK)
			if err != nil {
				log.Printf("[HybridSearch] Vector search failed: %v", err)
			} else {
				anchored := hr.anchorVectorResults(vectorResults)
				results = append(results, anchored...)
				log.Printf("[HybridSearch] Vector search: %d result(s)", len(anchored))
			}
		}
	}

	// Step 3: Merge and return top-K
	return hr.mergeResults(results, topK), nil
}

// directPageLookup checks whether the query explicitly names one or more pages
// (e.g. "what is on page 4?") and returns those PageIndex records.
func (hr *HybridRetriever) directPageLookup(query string) ([]SearchResult, bool) {
	db := database.GetDB()
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
				Score:       1.0, // exact match → highest confidence
				Source:      "page_index",
			})
		}
	}

	return results, len(results) > 0
}

// isSemanticQuery returns true when the query contains natural-language
// indicators that suggest a broad semantic search is appropriate.
func (hr *HybridRetriever) isSemanticQuery(query string) bool {
	indicators := []string{
		"explain", "describe", "what is", "how does", "why", "tell me",
		"summary", "overview", "information about", "details about",
	}
	lowerQuery := strings.ToLower(query)
	for _, ind := range indicators {
		if strings.Contains(lowerQuery, ind) {
			return true
		}
	}
	return false
}

// anchorVectorResults converts Milvus search results into SearchResults,
// enriching each with its PageIndex metadata (keywords, summary, offsets).
//
// FIX (original): used entity.SearchResult (wrong type from SDK's client package),
// panicked on empty data via unsafe type assertions, and needlessly parsed vector IDs
// to recover document/page info that is now returned directly by SearchVectors.
func (hr *HybridRetriever) anchorVectorResults(vectorResults []milvus.VectorSearchResult) []SearchResult {
	db := database.GetDB()
	var results []SearchResult

	for _, vr := range vectorResults {
		var pageIndex models.PageIndex
		err := db.Where("document_id = ? AND page_number = ?",
			vr.DocumentID, int(vr.PageNumber)).First(&pageIndex).Error

		if err != nil {
			// PageIndex missing — still surface the result with vector-DB data
			log.Printf("[Anchor] PageIndex not found doc=%s page=%d: %v",
				vr.DocumentID, vr.PageNumber, err)
			results = append(results, SearchResult{
				DocumentID:  vr.DocumentID,
				PageNumber:  int(vr.PageNumber),
				TextContent: vr.TextContent,
				Score:       vr.Score,
				Source:      "vector_db",
			})
			continue
		}

		results = append(results, SearchResult{
			DocumentID:  vr.DocumentID,
			PageNumber:  pageIndex.PageNumber,
			Summary:     pageIndex.Summary,
			Keywords:    pageIndex.Keywords,
			TextContent: vr.TextContent,
			OffsetStart: pageIndex.OffsetStart,
			OffsetEnd:   pageIndex.OffsetEnd,
			Score:       vr.Score,
			Source:      "vector_db",
		})
	}

	return results
}

// mergeResults deduplicates by (document_id, page_number), sorts by score
// descending, and trims to topK.
//
// FIX: replaced O(n²) bubble sort with sort.Slice.
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
		if len(merged) >= topK {
			break
		}
	}

	return merged
}

// --- Text processing helpers ---

func chunkText(text string, chunkSize int) []string {
	words := strings.Fields(text)
	var chunks []string
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
	// FIX: original had mixed tabs/spaces causing gofmt warnings
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
		if len(word) > 3 && !stopWords[word] {
			keywords[word] = true
		}
	}

	result := make([]string, 0, len(keywords))
	for w := range keywords {
		result = append(result, w)
	}
	return result
}

func generateSummary(text string) string {
	const maxWords = 50
	words := strings.Fields(text)
	if len(words) > maxWords {
		return strings.Join(words[:maxWords], " ") + "..."
	}
	return text
}

func countTables(text string) int {
	return strings.Count(text, "|") / 2
}
