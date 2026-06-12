# Enterprise RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** API built with FastAPI, FAISS, SentenceTransformers, and a pluggable in-house LLM backend.

---

## Architecture

```
S3 Documents
     │
     ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Ingestion  │───▶│  Embeddings  │───▶│  FAISS Index │
│  Pipeline   │    │  (Encoder)   │    │  (on disk)   │
└─────────────┘    └──────────────┘    └──────┬───────┘
                                              │
                                    ┌─────────▼────────┐
                          Query ───▶│   Retriever +    │
                                    │   Reranker       │
                                    └─────────┬────────┘
                                              │
                                    ┌─────────▼────────┐
                                    │  Prompt Builder  │
                                    └─────────┬────────┘
                                              │
                                    ┌─────────▼────────┐
                                    │   In-house LLM   │
                                    └─────────┬────────┘
                                              │
                                    ┌─────────▼────────┐
                                    │  Guardrails +    │
                                    │  Validator       │
                                    └─────────┬────────┘
                                              │
                                           Answer
```

---

## Project Structure

```
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Settings (pydantic-settings)
│   ├── ingestion/              # S3 loader, PDF/CSV parsers, chunker, tagger
│   ├── embeddings/             # Encoder, FAISS store, index builder
│   ├── retrieval/              # Retriever, MMR reranker
│   ├── generation/             # Prompt builder, LLM client, guardrails, validator
│   ├── prompts/                # Versioned prompt templates (v1, v2)
│   └── api/                   # Routes (/query, /ingest, /health), schemas, middleware
├── scripts/
│   ├── build_index.py          # One-time index build from S3
│   └── test_query.py           # Manual E2E query test
├── tests/                     # pytest test suite
├── data/
│   ├── raw/                   # Local S3 document cache (gitignored)
│   └── faiss_index/           # FAISS index files (gitignored)
├── .env                       # Environment variables
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd enterprise-rag-system
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env` and fill in your credentials:

```bash
cp .env .env.local
# Edit .env.local with your AWS keys, LLM endpoint, etc.
```

### 3. Build the FAISS index

```bash
python scripts/build_index.py --bucket my-bucket --prefix knowledge-base/
```

### 4. Run the API server

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Query the system

```bash
# Via the test script
python scripts/test_query.py --query "What is the refund policy?"

# Via curl
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "top_k": 5}'
```

---

## API Endpoints

| Method | Path              | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/health`         | Liveness probe                     |
| POST   | `/api/v1/query`   | RAG query — returns answer + sources |
| POST   | `/api/v1/ingest`  | Trigger S3 ingestion (async)       |

### POST `/api/v1/query`

**Request:**
```json
{
  "query": "What is the company refund policy?",
  "top_k": 5,
  "prompt_version": "v2"
}
```

**Response:**
```json
{
  "answer": "The refund policy allows returns within 30 days...",
  "sources": [
    {
      "chunk_id": "abc123",
      "text": "Returns are accepted within 30 days of purchase.",
      "source": "s3://my-bucket/policy.pdf",
      "score": 0.9231
    }
  ],
  "grounded": true,
  "overlap_score": 0.312
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
# Build
docker build -t enterprise-rag .

# Run
docker run -p 8000:8000 --env-file .env enterprise-rag
```

---

## Configuration Reference

| Variable                  | Default                                         | Description                     |
|---------------------------|-------------------------------------------------|---------------------------------|
| `AWS_ACCESS_KEY_ID`       | —                                               | AWS credentials                 |
| `AWS_SECRET_ACCESS_KEY`   | —                                               | AWS credentials                 |
| `S3_BUCKET_NAME`          | —                                               | Source document bucket          |
| `S3_PREFIX`               | `""`                                            | S3 key prefix / folder          |
| `EMBEDDING_MODEL_NAME`    | `sentence-transformers/all-MiniLM-L6-v2`        | HuggingFace model               |
| `EMBEDDING_DEVICE`        | `cpu`                                           | `cpu` or `cuda`                 |
| `FAISS_INDEX_PATH`        | `data/faiss_index/index.faiss`                  | Saved index location            |
| `FAISS_TOP_K`             | `5`                                             | Default retrieval top-k         |
| `LLM_ENDPOINT`            | `http://localhost:8080/generate`                | In-house LLM REST endpoint      |
| `LLM_API_KEY`             | —                                               | Bearer token for LLM            |
| `LLM_MAX_TOKENS`          | `1024`                                          | Max generation tokens           |
| `LLM_TEMPERATURE`         | `0.2`                                           | Sampling temperature            |
| `CHUNK_SIZE`              | `512`                                           | Max tokens per chunk            |
| `CHUNK_OVERLAP`           | `64`                                            | Overlap tokens between chunks   |
| `RATE_LIMIT_REQUESTS`     | `100`                                           | Max requests per window         |
| `RATE_LIMIT_WINDOW_SECONDS` | `60`                                          | Rate limit window in seconds    |

---

## License

MIT
