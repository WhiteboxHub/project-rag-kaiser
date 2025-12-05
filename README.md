# Kaiser RAG - Retrieval-Augmented Generation System

A complete RAG (Retrieval-Augmented Generation) system for querying Kaiser health insurance documents using embeddings and LLMs.

##  Architecture

```
PDFs (Data) → Ingestion Pipeline → Vector Store (Chroma) → RAG Pipeline → LLM → Answer
```

### Components

1. **Ingestion** (`app/ingestion/`)
   - `pipeline.py` - Orchestrates the ingestion workflow
   - `doc_loader.py` - Loads PDFs and extracts text
   - `chunker.py` - Splits text into chunks
   - `embedder.py` - Generates embeddings using OpenAI
   - `chroma_client.py` - Stores embeddings in local Chroma database

2. **RAG System** (`rag/`)
   - `retriever.py` - Queries vector store for relevant chunks
   - `generator.py` - Generates answers using LLM (GPT-4)
   - `query_pipeline.py` - Orchestrates retrieval + generation

3. **API** (`app/api/v1/`)
   - `router.py` - FastAPI endpoints
   - `schemas.py` - Request/response models

4. **Data**
   - `data/kaiser/` - Local PDF files (automatically ingested)
   - `data/embeddings/chroma/` - Vector store (persistent storage)

##  Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env
```

### 2. Ingest Documents

```bash
# Load PDFs into vector store
python ./scripts/run_ingestion.py
```

**Expected Output:**
```
{'chunks': 156, 'status': 'success', 'source': 'kaiser_principles'}
{'chunks': 896, 'status': 'success', 'source': 'kaiser_principles'}
{'chunks': 26, 'status': 'success', 'source': 'kaiser_principles'}
```

### 3. Test Locally

```bash
# Query the RAG system
python ./scripts/test_rag.py
```

### 4. Start API Server

```bash
# Run FastAPI server
uvicorn app.main:app --reload

# Open browser: http://localhost:8000/docs
```

## 📡 API Endpoints

### Health Check
```bash
GET /v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "RAG API is running"
}
```

### Query
```bash
POST /v1/query
Content-Type: application/json

{
  "question": "What are the principles of responsibility?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What are the principles of responsibility?",
  "answer": "Based on the retrieved documents, the principles of responsibility include...",
  "context": [
    "Kaiser Permanente's principles of responsibility include...",
    "..."
  ],
  "scores": [0.95, 0.92, ...],
  "num_chunks": 5
}
```

##  Project Structure

```
kaiser-rag/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── api/v1/                 # API routes & schemas
│   ├── core/                   # Configuration & logging
│   ├── ingestion/              # Document ingestion
│   └── schemas/                # Data models
├── rag/                        # RAG components
│   ├── retriever.py           # Vector store queries
│   ├── generator.py           # LLM response generation
│   └── query_pipeline.py      # Orchestration
├── scripts/
│   ├── run_ingestion.py       # Ingest documents
│   └── test_rag.py            # Test queries locally
├── data/
│   ├── kaiser/                # PDF documents
│   └── embeddings/chroma/     # Vector store
└── requirements.txt           # Dependencies
```

## ⚙️ Configuration

Edit `app/core/config.py` to customize:

```python
OPENAI_API_KEY: str                    # Your OpenAI API key
EMBEDDING_MODEL: str = "text-embedding-3-small"  # Embedding model
CHUNK_SIZE: int = 800                  # Text chunk size
CHUNK_OVERLAP: int = 100               # Overlap between chunks
```

##  How It Works

### Ingestion Flow
1. Load PDF from `data/kaiser/`
2. Extract text using pypdf
3. Split into chunks (800 tokens, 100 token overlap)
4. Generate embeddings (OpenAI `text-embedding-3-small`)
5. Store in Chroma (local SQLite)

### Query Flow
1. User asks a question
2. Embed question using OpenAI embeddings
3. Query Chroma for top-5 most relevant chunks
4. Pass question + context to GPT-4
5. Return generated answer with sources

##  Current Status

 **Completed:**
- Document ingestion (1,078 chunks indexed)
- Vector store (Chroma)
- RAG pipeline
- FastAPI endpoints

 **Future Enhancements:**
- Query reranking
- Multi-query expansion
- Response caching
- Evaluation metrics (RAGAS)
- Admin API for managing documents

##  Troubleshooting

**"No documents retrieved"**
- Ensure `run_ingestion.py` was executed successfully
- Check `data/embeddings/chroma/` directory exists

**"Import errors"**
- Verify virtual environment is activated
- Run `pip install -r requirements.txt`

**"OpenAI API errors"**
- Check `.env` file has valid `OPENAI_API_KEY`
- Verify API key has sufficient credits

##  License

Internal Use Only

