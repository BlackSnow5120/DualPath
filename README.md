# Page Index - Hybrid Retrieval System in Go

A production-ready hybrid retrieval system that combines deterministic Page Index (PostgreSQL/Redis) with probabilistic Vector DB (Milvus) for accurate and fast document retrieval.

## 🎯 Overview

This system implements a "Page Index" that sits alongside a traditional Vector DB to provide:

1. **Deterministic Lookup**: Fast, precise location-based searches (e.g., "What's on page 4?")
2. **Semantic Search**: Probabilistic vector-based searches for broad queries (e.g., "Explain the warranty policy")
3. **Hybrid Retrieval**: Combines both approaches for optimal results
4. **Context Merging**: Re-anchors vector results to their original document locations

## 🏗️ Architecture

### Dual Storage Layer

| Storage Type | Purpose | Query Type | Implementation |
|-------------|---------|-----------|----------------|
| **Page Index** (PostgreSQL) | Store document locations, page numbers, keywords | Direct lookup (exact matches) | GORM ORM |
| **Vector DB** (Milvus) | Store semantic embeddings | Semantic search (similarity) | Milvus SDK |
| **Cache** (Redis) | Fast retrieval of recent queries | Key-based lookup | go-redis |

### Processing Pipeline

When a document is uploaded:

```
┌─────────────┐
│  Document   │
└──────┬──────┘
       │
       ├──────────────────────────┐
       │                          │
       ▼                          ▼
┌──────────────┐          ┌──────────────┐
│ Vector       │          │ Index        │
│ Worker       │          │ Worker       │
│ (Goroutine)  │          │ (Goroutine)  │
└──────┬───────┘          └──────┬───────┘
       │                          │
       │                          │
       ▼                          ▼
┌──────────────┐          ┌──────────────┐
│ Embedding    │          │ Parse        │
│ Model        │          │ Structure    │
│             │          │ (pages,      │
│             │          │ headers)     │
└──────┬───────┘          └──────┬───────┘
       │                          │
       │                          │
       ▼                          ▼
┌──────────────┐          ┌──────────────┐
│ Milvus       │          │ PostgreSQL   │
│ (Vectors)    │          │ (Page Index) │
└──────────────┘          └──────────────┘
```

### Hybrid Search Flow

```
User Query: "What is on page 4?"
     │
     ▼
┌──────────────────────────────────┐
│ 1. Direct Page Lookup Check      │
│    Detects: "page 4"             │
│    Returns: Exact page data      │
│    Score: 1.0 (highest)          │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ 2. Semantic Search (if needed)   │
│    Embedding → Milvus Search    │
│    Results: Similar content      │
│    Score: 0.0-1.0               │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ 3. Context Merging              │
│    Re-anchor vector results     │
│    Deduplicate by doc/page      │
│    Sort by score                │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ 4. Return Combined Results       │
│    Page Index + Vector DB       │
└──────────────────────────────────┘
```

## 📁 Project Structure

```
DualPath/
├── main.go                 # Application entry point
├── go.mod                  # Go module definition
├── .env.example            # Environment variables template
├── Makefile                # Build and run commands
├── docker-compose.yml      # Infrastructure services
├── models/                 # Database models
│   └── index.go           # PageIndex, Document, Conversation
├── database/               # Database connections
│   ├── postgres.go        # PostgreSQL connection
│   └── redis.go           # Redis client
├── milvus/                # Vector database operations
│   └── client.go          # Milvus SDK integration
├── retrieval/             # Hybrid retrieval logic
│   └── hybrid.go          # Hybrid retrieval implementation
└── handlers/              # API handlers
    └── prompt.go          # Request/response handlers
```

## 🚀 Quick Start

### 1. Prerequisites

- Go 1.21 or higher
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+
- Milvus 2.2+

### 2. Start Infrastructure

```bash
# Start all services (PostgreSQL, Redis, Milvus)
make docker-up

# Check service status
make docker-logs
```

### 3. Install Dependencies

```bash
make install
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
nano .env
```

### 5. Run the Application

```bash
make run
```

Or simply:

```bash
make start
```

The API will be available at `http://localhost:8080`

## 🔌 API Endpoints

### Health Check
```bash
GET /health
```

### Upload Document
```bash
POST /api/v1/documents/upload
Content-Type: multipart/form-data

FormData:
- document_id: "doc_123"
- file: [binary data]
```

### Send Prompt
```bash
POST /api/v1/prompt
Content-Type: application/json

{
  "prompt": "What is on page 4?",
  "top_k": 5,
  "model": "gpt-4",
  "conversation_id": "optional_id"
}
```

Response:
```json
{
  "prompt": "What is on page 4?",
  "response": "On page 4, the document discusses the warranty policy...",
  "context": [
    {
      "document_id": "doc_123",
      "page_number": 4,
      "summary": "Warranty policy details...",
      "score": 1.0,
      "source": "page_index"
    }
  ],
  "conversation_id": "uuid-here",
  "timestamp": "2026-03-15T01:30:00Z",
  "sources": [
    {"document_id": "doc_123", "page_number": 4}
  ]
}
```

### Search Documents
```bash
GET /api/v1/search?q=payment+terms&top_k=5
```

### Get Conversation
```bash
GET /api/v1/conversations/:id
```

### List Conversations
```bash
GET /api/v1/conversations
```

## 📊 Data Schema

### Page Index (PostgreSQL)

```go
type PageIndex struct {
    ID          uint
    DocumentID  string
    PageNumber  int
    Keywords    string      // Comma-separated keywords
    Summary     string      // Extracted summary
    OffsetStart int64       // Byte offset start
    OffsetEnd   int64       // Byte offset end
    Metadata    string      // JSON (headers, tables)
    CreatedAt   time.Time
    UpdatedAt   time.Time
}
```

### Document (PostgreSQL)

```go
type Document struct {
    ID          uint
    DocumentID  string      // Unique identifier
    FileName    string
    ContentType string
    FilePath    string
    TotalPages  int
    FileSize    int64
    IsProcessed bool
    CreatedAt   time.Time
    UpdatedAt   time.Time
}
```

### Vector Collection (Milvus)

- **Collection Name**: `document_vectors`
- **Dimension**: 1536 (OpenAI embeddings)
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Distance Metric**: Cosine Similarity

### Conversation Storage

Stored in PostgreSQL and synchronized to:
1. **Vector DB**: For semantic retrieval of past conversations
2. **Redis**: For fast caching (24-hour TTL)

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | PostgreSQL host | localhost |
| `DB_PORT` | PostgreSQL port | 5432 |
| `DB_USER` | PostgreSQL user | postgres |
| `DB_PASSWORD` | PostgreSQL password | - |
| `DB_NAME` | Database name | dualpath |
| `REDIS_ADDR` | Redis address | localhost:6379 |
| `REDIS_PASSWORD` | Redis password | - |
| `REDIS_DB` | Redis DB number | 0 |
| `MILVUS_ADDR` | Milvus address | localhost:19530 |
| `PORT` | API server port | 8080 |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |

## 🧪 Testing

```bash
# Run all tests
make test

# Test specific package
go test ./retrieval -v
```

## 📝 Usage Examples

### Example 1: Direct Page Lookup

Request: "What is on page 4?"

Response:
```json
{
  "prompt": "What is on page 4?",
  "response": "Page 4 contains the warranty policy information...",
  "context": [
    {
      "document_id": "doc_123",
      "page_number": 4,
      "summary": "Warranty policy covers...",
      "score": 1.0,
      "source": "page_index"
    }
  ]
}
```

### Example 2: Semantic Search

Request: "Explain the warranty policy"

Response:
```json
{
  "prompt": "Explain the warranty policy",
  "response": "The warranty policy covers...",
  "context": [
    {
      "document_id": "doc_456",
      "page_number": 7,
      "summary": "Warranty details section...",
      "score": 0.92,
      "source": "vector_db"
    },
    {
      "document_id": "doc_789",
      "page_number": 12,
      "summary": "Related warranty terms...",
      "score": 0.87,
      "source": "vector_db"
    }
  ]
}
```

### Example 3: Hybrid Search (Mixed)

Request: "What are the payment terms mentioned on page 4?"

Response combines direct page lookup + semantic search results.

## 🔍 How It Works

### 1. Initialization Pipeline

When processing a document:

```go
// Two parallel goroutines
go vectorWorker()   // Embedding → Milvus
go indexWorker()    // Structured metadata → PostgreSQL
```

### 2. Hybrid Retrieval Logic

```go
// Step 1: Check for direct page mentions
if containsPageNumber(query) {
    return index.Search(pageNumber)
}

// Step 2: Semantic search for broad queries
if isSemanticQuery(query) {
    return milvus.Search(embedding(query))
}

// Step 3: Merge results
return mergeAndDeduplicate(results)
```

### 3. Context Serialization

Context is stored as JSON in multiple locations:
- **PostgreSQL**: Long-term storage
- **Redis**: Fast cache (24h)
- **Vector DB**: For future retrieval

## 🐳 Docker Services

### PostgreSQL
- Port: 5432
- Database: dualpath
- User: postgres

### Redis
- Port: 6379
- Data persistence: enabled

### Milvus
- Ports: 19530 (gRPC), 9091 (HTTP)
- Standalone mode
- Embedded MinIO

## 🛠️ Development

### Add a New Model

```go
// models/new_model.go
type NewModel struct {
    ID   uint `gorm:"primary_key"`
    Name string
}
```

### Add a New Endpoint

```go
// main.go
v1.GET("/new-endpoint", handlers.NewEndpoint)

// handlers/new.go
func NewEndpoint(c *gin.Context) {
    c.JSON(200, gin.H{"message": "response"})
}
```

## 🚀 Production Deployment

### Build Binary

```bash
make build
```

### Run Binary

```bash
./dualpath
```

### Docker Deployment

```bash
docker build -t dualpath .
docker run -p 8080:8080 --env-file .env dualpath
```

## 📚 Future Enhancements

- [ ] Real embedding API integration (OpenAI, Cohere)
- [ ] LLM API integration (GPT-4, Claude)
- [ ] Multi-document processing
- [ ] PDF/Excel/parser
- [ ] WebSocket support for real-time updates
- [ ] OAuth2 authentication
- [ ] Rate limiting
- [ ] Monitoring and metrics (Prometheus)
- [ ] Distributed tracing (Jaeger)
- [ ] Caching optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Commit and push
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 👥 Authors

- Your Name - Initial work

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

Built with Go, Gin, PostgreSQL, Redis, and Milvus.
