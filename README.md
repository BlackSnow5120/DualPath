# DualPath — Hybrid Retrieval System (Supabase Edition)

A production-ready hybrid retrieval system built with **Go** and **Supabase**. It consolidates deterministic Page Indexing and probabilistic Vector Search into a single, high-performance PostgreSQL instance using **pgvector**.

## 🎯 Overview

This system implements a unified "Page Index" that provides:

1. **Deterministic Lookup**: Fast, precise location-based searches (e.g., "What's on page 4?")
2. **Semantic Search**: Probabilistic vector-based searches via **pgvector** (e.g., "Explain the warranty policy")
3. **Hybrid Retrieval**: Combines both approaches for optimal context retrieval.
4. **All-in-One Storage**: No separate Vector DB or Cache required; everything lives in Supabase.

## 🏗️ Architecture

### Consolidated Storage Layer

| Storage Type | Purpose | Query Type | Implementation |
|:---|:---|:---|:---|
| **Relational Data** | Store document metadata, page numbers, keywords | Direct lookup (exact matches) | Supabase (GORM) |
| **Vector Data** | Store semantic embeddings for each page | Semantic search (similarity) | **pgvector** (<=> operator) |
| **History** | Store conversation context and source IDs | Metadata lookup | PostgreSQL |

### Processing Pipeline

When a document is uploaded, it is processed in parallel to extract text, keywords, and generate semantic embeddings.

```
┌─────────────┐
│  Document   │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ Single Worker│
│ (Goroutine)  │
└──────┬───────┘
       │
       ├──────────────────────────┐
       │                          │
       ▼                          ▼
┌──────────────┐          ┌──────────────┐
│ Embedding    │          │ Structural   │
│ Model        │          │ Parsing      │
│ (OpenAI)     │          │ (Keywords)   │
└──────┬───────┘          └──────┬───────┘
       │                          │
       └────────────┬─────────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ Supabase Postgres  │
          │ (Relational + Vec) │
          └────────────────────┘
```

## 📁 Project Structure

```
DualPath/
├── main.go                 # Application entry point
├── go.mod                  # Go module definition
├── .env.example            # Environment variables template
├── models/                 # Database models
│   └── index.go           # PageIndex, Document, Conversation
├── database/               # Database connections
│   └── postgres.go        # Supabase PostgreSQL connection
├── retrieval/             # Hybrid retrieval logic
│   └── hybrid.go          # Hybrid retrieval + pgvector search
└── handlers/              # API handlers
    └── prompt.go          # Request/response handlers
```

## 🚀 Quick Start

### 1. Prerequisites

- Go 1.22+
- A Supabase Project (with PostgreSQL 15+)
- OpenAI/Anthropic API Keys

### 2. Configure Supabase

Run the following in your Supabase SQL Editor:

```sql
-- Enable vector support
CREATE EXTENSION IF NOT EXISTS vector;

-- Add optimized index for cosine similarity
CREATE INDEX ON page_indexes USING hnsw (embedding vector_cosine_ops);
```

### 3. Configure Environment

```bash
cp .env.example .env
# Update DATABASE_URL with your Supabase connection string
```

### 4. Run the Application

```bash
go run main.go
```

The API will be available at `http://localhost:8080`

## 🔌 API Endpoints

### Health Check
`GET /health` - Reports Supabase connection status.

### Upload Document
`POST /api/v1/documents/upload`
- `document_id`: "doc_123"
- `file`: [binary data]

### Send Prompt
`POST /api/v1/prompt`
```json
{
  "prompt": "What is on page 4?",
  "top_k": 5,
  "model": "gpt-4"
}
```

## 📊 Data Schema

### Page Index (Supabase)

```go
type PageIndex struct {
    ID          uint
    DocumentID  string
    PageNumber  int
    Keywords    string
    Summary     string
    Embedding   pgvector.Vector // Stored as vector(1536)
    Metadata    string          // JSON (headers, tables)
    CreatedAt   time.Time
}
```

## 🔧 Configuration

| Variable | Description |
|:---|:---|
| `DATABASE_URL` | Supabase Postgres Connection String |
| `OPENAI_API_KEY` | OpenAI API key (for Embeddings + GPT) |
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude) |

## 🧪 Testing

```bash
# Run all tests
go test ./... -v
```

## 🔍 How It Works

### Hybrid Retrieval Logic

1. **Direct Lookup**: The system uses regex to detect explicit page mentions (e.g., "page 5") and fetches them directly from Postgres with a perfect score (1.0).
2. **Semantic Search**: If no page is mentioned, it generates an embedding for the query and uses the `<=>` (cosine distance) operator in SQL to find matching page segments.
3. **Merge & Rank**: Results from both paths are merged, deduplicated, and ranked before being sent to the LLM as context.

---

Built with Go, Gin, and Supabase.
