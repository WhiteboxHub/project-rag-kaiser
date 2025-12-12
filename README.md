# Kaiser RAG - Retrieval-Augmented Generation System

A complete RAG (Retrieval-Augmented Generation) system for querying Kaiser health insurance documents using local transformer embeddings and LLMs.

##  Architecture

```
PDFs (Data) → Ingestion Pipeline → Vector Store (Chroma) → RAG Pipeline → LLM → Answer
```

### Components

1. **Ingestion** (`app/ingestion/`)
   - `pipeline.py` - Orchestrates the ingestion workflow
   - `doc_loader.py` - Loads PDFs with **page-level tracking**
   - `metadata_chunker.py` - Splits text while **extracting metadata** (page numbers, chapters, sections)
   - `embedder.py` - Generates embeddings using **Sentence Transformers** (`all-MiniLM-L6-v2`)
   - `chroma_client.py` - Stores embeddings with **rich metadata** in local Chroma database

2. **RAG System** (`rag/`)
   - `retriever.py` - **Hybrid search**: semantic similarity + metadata filtering
   - `generator.py` - Generates answers using LLM (GPT-4)
   - `query_pipeline.py` - Orchestrates retrieval + generation

3. **User Interface**
   - `streamlit_app.py` - Interactive web interface with **metadata display** (source file, page, chapter)

4. **API** (`app/api/v1/`)
   - `router.py` - FastAPI endpoints
   - `schemas.py` - Request/response models

5. **Data**
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
# Note: If changing embedding models, it's recommended to clear 'data/embeddings/chroma' first
python ./scripts/run_ingestion.py
```

### 3. Run Application

**Option A: Streamlit UI (Recommended)**
```bash
streamlit run streamlit_app.py
```
Open your browser at the URL shown (typically http://localhost:8501).

**Option B: Test Locally via CLI**
```bash
# Query the RAG system
python ./scripts/test_rag.py --default
# Or ask a custom question
python ./scripts/test_rag.py --question "What are the benefits?"
```

**Option C: Start API Server**
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

##  Project Structure

```
kaiser-rag/
├── streamlit_app.py            # Streamlit User Interface [NEW]
├── app/
│   ├── main.py                 # FastAPI application
│   ├── api/v1/                 # API routes & schemas
│   ├── core/                   # Configuration & logging
│   ├── ingestion/              # Document ingestion
│   └── schemas/                # Data models
├── rag/                        # RAG components
│   ├── retriever.py            # Vector store queries
│   ├── generator.py            # LLM response generation
│   └── query_pipeline.py       # Orchestration
├── scripts/
│   ├── run_ingestion.py        # Ingest documents
│   └── test_rag.py             # Test queries locally
├── data/
│   ├── kaiser/                 # PDF documents
│   └── embeddings/chroma/      # Vector store
└── requirements.txt            # Dependencies
```

## ⚙️ Configuration

Edit `app/core/config.py` to customize:

```python
OPENAI_API_KEY: str                    # Your OpenAI API key
# Embedding model is configured in code to use 'sentence-transformers/all-MiniLM-L6-v2'
```

##  How It Works

### Ingestion Flow (with Metadata Extraction)
1. Load PDF from `data/kaiser/` **with page-level tracking**
2. Extract text using pypdf, **preserving page numbers**
3. Split into chunks while **extracting metadata**:
   - Page numbers (e.g., page 303)
   - Chapter numbers (e.g., "Chapter 12")
   - Section titles
4. Generate embeddings (HuggingFace `sentence-transformers/all-MiniLM-L6-v2`)
5. Store in Chroma with **full metadata** for each chunk

### Query Flow (with Hybrid Search)
1. User asks a question
2. **Extract metadata filters** from query (e.g., "Chapter 12" → filter `chapter="12"`)
3. Embed question using the same transformer model
4. **Hybrid search**: Retrieve top chunks using:
   - 70% semantic similarity score
   - 30% metadata match bonus (chapter, page, source file)
5. Pass question + context to GPT-4
6. Return answer with **source metadata** (file, page, chapter)

##  Key Features

✅ **Metadata-Enhanced Retrieval**
- Automatically extracts page numbers, chapters, and sections from PDFs
- Hybrid search combines semantic + metadata matching
- Handles structural queries like "what is in chapter 12"

✅ **Smart Query Understanding**
- Detects chapter/page references in queries
- Filters results to relevant sections automatically
- Boosts relevance for matching source files

✅ **Transparent Source Attribution**
- Every result shows source file name
- Page numbers for easy verification
- Chapter information when available

##  Current Status

 **Completed:**
- Document ingestion working
- Vector store (Chroma)
- RAG pipeline
- **Streamlit UI**
- FastAPI endpoints

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

