# Page Index - Quick Start Guide

## 🚀 5-Minute Setup

### 1. Start Infrastructure (2 min)

```bash
cd ~/Desktop/DualPath
make docker-up
```

Wait for all containers to be healthy.

### 2. Configure Environment (1 min)

```bash
cp .env.example .env
nano .env
```

Edit any necessary credentials.

### 3. Run the Application (2 min)

```bash
make start
```

The API will be available at http://localhost:8080

## 🧪 Test It Out

### Upload a Document

```bash
curl -X POST http://localhost:8080/api/v1/documents/upload \
  -F "document_id=test_doc_1" \
  -F "file=@example.pdf"
```

### Send a Prompt

```bash
curl -X POST http://localhost:8080/api/v1/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is on page 1?","top_k":5}'
```

### Search Documents

```bash
curl "http://localhost:8080/api/v1/search?q=warranty&top_k=5"
```

## 📊 What This System Does

1. **Deterministic Page Lookup**: Exact page queries → PostgreSQL (fast)
2. **Semantic Search**: Broad concept queries → Milvus (smart)
3. **Hybrid Results**: Combines both for accuracy
4. **Context Merging**: Re-anchors results to original docs
5. **Conversation Storage**: Saves prompts + responses in Vector DB for retrieval

## 🔑 Key Features

- ✅ Parallel document processing (Vector Worker + Index Worker)
- ✅ Hybrid retrieval (Page Index + Vector DB)
- ✅ Context caching in Redis
- ✅ Conversation history stored in PostgreSQL + Vector DB
- ✅ Easy REST API with Gin framework
- ✅ Docker Compose for easy infrastructure setup

## 📁 Project Files

| File | Purpose |
|------|---------|
| `main.go` | API server and routes |
| `models/index.go` | Database models |
| `database/postgres.go` | PostgreSQL connection |
| `database/redis.go` | Redis client |
| `milvus/client.go` | Vector DB operations |
| `retrieval/hybrid.go` | Hybrid search logic with parallel workers |
| `handlers/prompt.go` | API endpoints |

## 🛠️ Common Commands

```bash
make install    # Install dependencies
make run        # Run the application
make build      # Build binary
make test       # Run tests
make docker-up  # Start infrastructure
make docker-down # Stop infrastructure
```

## 📖 Documentation

See full [README.md](README.md) for detailed documentation.

## 💡 Example Outputs

### Direct Page Lookup

**Request**: "What is on page 4?"

**Response**: Directly from Page Index (score: 1.0)

### Semantic Search

**Request**: "Explain the warranty policy"

**Response**: From Vector DB with top 5 similar chunks (score: 0.0-1.0)

### Mixed Query

**Request**: "What payment terms are on page 7?"

**Response**: Page 7 content + related semantic results

---

**Questions?** Check the full README or open an issue on GitHub.
