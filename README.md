# Production Hybrid RAG Pipeline with Source Citations

**SkillCraft Technology — AI Engineering Internship (Task 1)**

An enterprise-ready Hybrid Retrieval-Augmented Generation (RAG) system integrating dense semantic vector search (ChromaDB + Sentence Transformers) and sparse keyword retrieval (BM25 Okapi) fused with Reciprocal Rank Fusion (RRF), cross-encoder re-ranking (`BAAI/bge-reranker-base`), and grounded response generation with strict source citations via the Google Gemini API.

---

## Architecture Overview

```text
                                 ┌────────────────────────┐
                                 │   User Query (React)   │
                                 └───────────┬────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  FastAPI Backend Gateway  │
                               └─────────────┬─────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
       ┌──────────────────────────┐                    ┌──────────────────────────┐
       │   Dense Vector Search    │                    │   Sparse Keyword Search  │
       │ ChromaDB + MiniLM-L6-v2  │                    │        BM25 Okapi        │
       └─────────────┬────────────┘                    └─────────────┬────────────┘
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Reciprocal Rank Fusion    │
                               │ (RRF Score Combination)   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   Cross-Encoder Reranker  │
                               │  (BAAI/bge-reranker-base) │
                               └─────────────┬─────────────┘
                                             │
                                             ▼ Top N Passages
                               ┌───────────────────────────┐
                               │     Google Gemini API     │
                               │  (Strict Source Citation) │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                                 ┌────────────────────────┐
                                 │ Cited Response (React) │
                                 └────────────────────────┘
```

---

## Phase 3: Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

### 1. Dense Semantic Retrieval
- **Query Vectorization:** Queries are encoded into 384-dimensional $L_2$-normalized dense vectors using `sentence-transformers/all-MiniLM-L6-v2` loaded through Hugging Face `AutoModel` and `AutoTokenizer` on CPU.
- **Similarity Computation:** Evaluated against ChromaDB collections using cosine distance space ($\text{Cosine Distance} = 1 - \text{Cosine Similarity}$).
- **Output:** Returns top candidate chunks ranked by cosine similarity with page citations and metadata.

### 2. BM25 Lexical Retrieval
- **Tokenization:** Query text is split into lowercase alphanumeric tokens, preserving technical jargon, numbers, and acronyms (`TCP`, `UDP`, `B-tree`).
- **Scoring:** Scored via `BM25Okapi` ($k_1=1.5, b=0.75$) with positive IDF floor smoothing ($\ge 0.25$) to handle small corpora and term frequency saturation.
- **Output:** Returns top candidate chunks ranked by lexical score with page citations and metadata.

### 3. Reciprocal Rank Fusion (RRF) Formula & Pipeline
Reciprocal Rank Fusion combines ranked lists from disparate retrieval mechanisms without requiring score calibration or normalization:

$$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$$

Where:
- $M = \{\text{dense}, \text{bm25}\}$: The set of independent retrieval systems.
- $\text{rank}_m(d)$: The 1-indexed ranking position of document chunk $d$ in system $m$ ($1, 2, 3, \dots$).
- $k$: The smoothing constant (configurable, default $k = 60$). It controls the score penalty decay between successive ranks.
- **Single-Source Matches:** If chunk $d$ appears in only one retriever (e.g. found in BM25 but not in top dense candidates), it receives its reciprocal score from that system only ($\frac{1}{k + \text{rank}_{\text{bm25}}(d)}$) while preserving `dense_rank = null`.
- **Deduplication & Sorting:** Unique chunk IDs are accumulated into a fused candidate pool, scored via RRF, and sorted in descending order of RRF score with deterministic tie-breaking.

---

## Technology Stack

| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend** | React.js + Vite | React 18, Vite 5.4 | Interactive dashboard, document manager, index monitor, retrieval comparator |
| **Backend** | Python + FastAPI | Python 3.11.9, FastAPI 0.115+ | High-performance asynchronous REST API Gateway |
| **PDF Extraction**| PyMuPDF | 2.1 | Page-level text extraction & layout preservation |
| **Tokenizer** | Subword BPE / Tiktoken | BPE Encoding | Subword token estimation (~500 target, 75 overlap) |
| **Dense Embeddings** | HuggingFace Transformers | 5.23.0 | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU) |
| **Vector DB** | ChromaDB | 1.5.9 | Persistent local dense vector store (`data/chroma_db/`) |
| **Sparse Index** | `rank-bm25` | 0.2.2 | BM25Okapi lexical matching with persistent JSON corpus |
| **Hybrid Fusion** | Reciprocal Rank Fusion | Algorithmic ($k=60$) | Multi-channel rank combination across dense and sparse systems |
| **Reranker** | BGE Reranker | 0.1.0+ | `BAAI/bge-reranker-base` cross-encoder sequence classification (CPU) |
| **LLM Engine** | Google Gemini API | `google-genai` 3.4+ | Grounded answer generation with source bracket citations |

---

## Directory Structure

```text
Skill Craft Technology -  AI - Task 1/
├── .env.example                         # Environment variable template (placeholders only)
├── .gitignore                           # Excludes venv, node_modules, keys, caches, indices, PDFs
├── README.md                            # Comprehensive project guide
│
├── documents/                           # Stored raw PDF knowledge documents (.gitkeep tracked)
├── data/                                # Persistent storage for indices, catalogs & benchmarks
│   ├── .gitkeep                         # Preserves data directory structure
│   ├── processed_documents/             # Ingested document metadata & chunk JSON files (.gitkeep tracked)
│   ├── chroma_db/                       # ChromaDB persistent collection files (gitignored)
│   ├── bm25_index/                      # BM25 persistent JSON corpus (gitignored)
│   └── evaluation/                      # RAG evaluation benchmark datasets & baseline reports
│       ├── dataset.json                 # Ground-truth evaluation questions & relevance judgments
│       └── benchmark_baseline_report.json # Reference baseline performance report
│
├── backend/                             # FastAPI application service
│   ├── requirements.txt                 # Pinned Python dependencies
│   ├── .venv/                           # Python 3.11.9 virtual environment (gitignored)
│   ├── scripts/                         # Standalone diagnostic, benchmark & test runner utilities
│   │   ├── run_unit_tests.py            # Phase 6 & regression test runner script
│   │   ├── run_evaluation.py            # CLI benchmark runner across all 6 retrieval configurations
│   │   └── ...                          # Diagnostics & corpus inspection scripts
│   ├── tests/
│   │   ├── test_phase2a_ingestion.py    # 8 automated ingestion & chunking test suites
│   │   ├── test_phase2b_indexing.py     # 12 automated dense vector & BM25 test suites
│   │   ├── test_phase3_retrieval.py     # 11 automated dense, BM25, and RRF test suites
│   │   ├── test_phase4_reranking.py     # 12 automated BGE cross-encoder reranking test suites
│   │   ├── test_phase5_generation.py    # 15 automated Gemini grounded generation & citation tests
│   │   └── test_phase6_evaluation.py    # 16 automated IR metrics, ranking & benchmark tests
│   └── app/
│       ├── __init__.py
│       ├── main.py                      # FastAPI app entry point, CORS & router registration
│       ├── core/
│       │   ├── __init__.py
│       │   └── config.py                # Pydantic BaseSettings, paths, upload size limits
│       ├── models/
│       │   ├── __init__.py
│       │   ├── health.py                # System health schema
│       │   ├── document.py              # ChunkMetadata, DocumentInfo, UploadResponse schemas
│       │   ├── index.py                 # IndexStatusResponse, IndexStatsResponse, Rebuild schemas
│       │   ├── retrieval.py             # RetrievalRequest, RetrievedChunk, RetrievalResponse schemas
│       │   ├── qa.py                    # QARequest, CitationSource, QAResponse schemas
│       │   └── evaluation.py            # EvaluationDataset, Question, Metrics & Report schemas
│       ├── services/
│       │   ├── __init__.py
│       │   ├── text_cleaner.py          # Whitespace & paragraph normalization
│       │   ├── document_loader.py       # PyMuPDF PDF validator & page text extractor
│       │   ├── text_splitter.py         # Sentence-aware subword token chunker
│       │   ├── storage_service.py       # Disk persistence & SHA-256 duplicate detection
│       │   ├── embedding_service.py     # HuggingFace MiniLM-L6-v2 CPU embedding inference
│       │   ├── vector_store.py          # ChromaDB PersistentClient vector store controller
│       │   ├── bm25_service.py          # BM25Okapi lexical inverted index service
│       │   ├── index_manager.py         # Dual-store synchronization & rebuild orchestrator
│       │   ├── retrieval_service.py     # Dense, BM25, and Hybrid RRF retrieval service
│       │   ├── reranker_service.py      # BAAI/bge-reranker-base cross-encoder inference service
│       │   ├── generation_service.py    # Google Gemini grounded generation & citation validator
│       │   └── evaluation_service.py    # Multi-configuration IR evaluator, nDCG, MRR & latency profiler
│       └── api/
│           ├── __init__.py
│           ├── routes_health.py         # GET /api/v1/health
│           ├── routes_documents.py      # POST upload, GET list, GET detail, DELETE doc
│           ├── routes_index.py          # GET status, GET stats, POST rebuild
│           ├── routes_retrieval.py      # POST /api/v1/retrieval/search
│           ├── routes_qa.py             # POST /api/v1/qa/ask (End-to-End Grounded Q&A)
│           └── routes_evaluation.py     # POST run, GET results, GET dataset, POST add
│
└── frontend/                            # React.js + Vite presentation layer
    ├── package.json                     # Frontend dependencies (React, Lucide icons)
    ├── package-lock.json                # Lockfile for deterministic builds
    ├── vite.config.js                   # Vite dev server & proxy settings (/api -> :8000)
    ├── index.html                       # HTML5 template
    └── src/
        ├── App.jsx                      # Main dashboard layout with live metrics & tab navigation
        ├── main.jsx                     # Vite mounting point
        ├── index.css                    # Professional dark theme styling
        ├── components/
        │   ├── Header.jsx               # Navigation bar & live backend status badge
        │   ├── PipelineOverview.jsx     # Visual architecture diagram of hybrid search
        │   ├── StatusCard.jsx           # Component status & technology cards
        │   ├── DocumentManager.jsx      # PDF drag-and-drop uploader, library table, chunk viewer
        │   ├── IndexManager.jsx         # Dual-index statistics, sync state & rebuild control
        │   ├── QuestionAnswer.jsx       # Grounded Q&A with clickable citation pills
        │   ├── RetrievalComparison.jsx  # Interactive search, strategy selector & chunk comparison
        │   └── EvaluationDashboard.jsx  # Evaluation metrics matrix, K-cutoff selector & latency profile
        └── services/
            └── api.js                   # Backend API client for health, documents, index, retrieval, QA & eval
```

---

## API Endpoints Reference

| Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Root info endpoint returning service metadata, version, and route links | 200 OK |
| `GET` | `/api/v1/health` | Live diagnostic status and total document count | 200 OK |
| `POST`| `/api/v1/documents/upload` | Validates, extracts, chunks, and incrementally indexes a PDF | 201 Created |
| `GET` | `/api/v1/documents` | Lists all ingested documents with extraction metrics | 200 OK |
| `GET` | `/api/v1/documents/{id}` | Returns document details, page breakdown, and all chunk texts | 200 OK / 404 |
| `DELETE`| `/api/v1/documents/{id}` | Permanently deletes raw PDF, chunks, and removes from both indices | 200 OK / 404 |
| `GET` | `/api/v1/index/status` | Returns orchestrator operational status, last action, and catalog totals | 200 OK |
| `GET` | `/api/v1/index/stats` | Returns comprehensive metrics for ChromaDB collection and BM25 index | 200 OK |
| `POST`| `/api/v1/index/rebuild` | Resets and rebuilds ChromaDB and BM25 from persistent JSON documents | 200 OK / 500 |
| `POST`| `/api/v1/retrieval/search` | Executes search across BM25, Dense, or Hybrid RRF strategies | 200 OK / 400 |
| `POST`| `/api/v1/qa/ask` | Grounded question answering with strict source citation attribution | 200 OK / 400 |
| `POST`| `/api/v1/evaluation/run` | Executes evaluation across all 6 retrieval configurations | 200 OK / 500 |
| `GET` | `/api/v1/evaluation/results` | Retrieves latest cached evaluation report and metrics | 200 OK / 404 |
| `GET` | `/api/v1/evaluation/dataset` | Retrieves benchmark questions and ground-truth judgments | 200 OK |
| `POST`| `/api/v1/evaluation/dataset/add` | Appends a new verified or candidate question to benchmark | 201 Created |

### Retrieval & Reranking API Example

**Request (with BGE Cross-Encoder Reranking):**
```json
POST /api/v1/retrieval/search
Content-Type: application/json

{
  "query": "How does self-attention mechanism work in transformers?",
  "retrieval_method": "hybrid",
  "top_k": 5,
  "rrf_k": 60,
  "rerank": true,
  "reranker_top_k": 3,
  "reranker_batch_size": 16
}
```

**Response:**
```json
{
  "query": "How does self-attention mechanism work in transformers?",
  "retrieval_method": "hybrid",
  "total_results": 3,
  "results": [
    {
      "rank": 1,
      "original_rank": 2,
      "rerank_rank": 1,
      "reranker_score": 0.8942,
      "chunk_id": "doc1_p1_c0",
      "document_id": "b78912c4-...",
      "document_name": "attention_is_all_you_need.pdf",
      "page_number": 1,
      "chunk_index": 0,
      "chunk_text": "An attention function can be described as mapping a query and a set of key-value pairs...",
      "token_count": 482,
      "score": 0.032522,
      "dense_score": 0.7412,
      "dense_rank": 2,
      "bm25_score": 2.4510,
      "bm25_rank": 1,
      "rrf_score": 0.032522
    }
  ],
  "execution_time_ms": 48.25,
  "embedding_time_ms": 15.30,
  "reranking_time_ms": 22.40,
  "dense_candidates_count": 20,
  "bm25_candidates_count": 20,
  "rrf_k": 60,
  "rerank_enabled": true,
  "reranker_model": "BAAI/bge-reranker-base"
}
```

### Grounded Q&A API Example (`POST /api/v1/qa/ask`)

**Request:**
```json
POST /api/v1/qa/ask
Content-Type: application/json

{
  "query": "How does self-attention work in the Transformer architecture?",
  "retrieval_method": "hybrid",
  "top_k": 5,
  "rerank": true,
  "reranker_top_k": 3,
  "temperature": 0.2
}
```

**Response:**
```json
{
  "query": "How does self-attention work in the Transformer architecture?",
  "answer": "The Transformer model relies entirely on a self-attention mechanism to compute representations without recurrent sequence-aligned RNNs [1]. Specifically, multi-head attention projects queries, keys, and values into multiple subspaces, enabling the model to attend to information from different positions simultaneously [2].",
  "retrieval_method": "hybrid",
  "rerank_enabled": true,
  "citations": [1, 2],
  "invalid_citations": [],
  "sources": [
    {
      "citation_id": 1,
      "citation_label": "[1]",
      "document_id": "doc_attn_123",
      "document_name": "attention_is_all_you_need.pdf",
      "page_number": 1,
      "chunk_id": "doc_attn_p1_c0",
      "chunk_index": 0,
      "chunk_text": "The Transformer is the first transduction model relying entirely on self-attention...",
      "token_count": 482,
      "retrieval_rank": 2,
      "rerank_rank": 1,
      "score": 0.032522,
      "dense_score": 0.7412,
      "dense_rank": 2,
      "bm25_score": 2.4510,
      "bm25_rank": 1,
      "rrf_score": 0.032522,
      "reranker_score": 0.8942
    },
    {
      "citation_id": 2,
      "citation_label": "[2]",
      "document_id": "doc_attn_123",
      "document_name": "attention_is_all_you_need.pdf",
      "page_number": 3,
      "chunk_id": "doc_attn_p3_c1",
      "chunk_index": 1,
      "chunk_text": "Multi-head attention allows the model to jointly attend to information...",
      "token_count": 468,
      "retrieval_rank": 1,
      "rerank_rank": 2,
      "score": 0.032115,
      "dense_score": 0.7620,
      "dense_rank": 1,
      "bm25_score": 1.9540,
      "bm25_rank": 2,
      "rrf_score": 0.032115,
      "reranker_score": 0.8521
    }
  ],
  "has_sufficient_context": true,
  "total_execution_time_ms": 1148.50,
  "retrieval_time_ms": 32.10,
  "embedding_time_ms": 14.80,
  "reranking_time_ms": 21.20,
  "generation_time_ms": 1095.20,
  "model_name": "gemini-2.5-flash"
}
```

---

## Why Cross-Encoder Reranking Follows Hybrid Retrieval

In high-accuracy production RAG pipelines, retrieval is structured as a **multi-stage coarse-to-fine funnel**:

1. **Stage 1 (Coarse Candidate Retrieval — High Recall, Low Latency):**
   - **BM25 Lexical Search:** Catches exact keywords, terminology, IDs, and acronyms with near-zero latency (<1 ms).
   - **Dense Bi-Encoder Vectors (`all-MiniLM-L6-v2`):** Generates independent embeddings for query and document chunks, measuring semantic proximity via cosine similarity. Fast and scalable via vector indexes, but misses fine-grained token-level cross-attention interactions.
2. **Stage 2 (Candidate Fusion — Reciprocal Rank Fusion):**
   - Combines disparate rank distributions into an equitable candidate pool without requiring delicate score normalization.
3. **Stage 3 (Precision Reranking — Cross-Encoder `bge-reranker-base`):**
   - Concatenates query and document into a single joint sequence: `[CLS] Query [SEP] Chunk [SEP]`.
   - Full all-to-all cross-attention allows every word in the query to attend directly to every word in the candidate chunk.
   - Outputs a calibrated sigmoid confidence score $P(\text{Relevant} \mid Q, D) \in [0, 1]$.
   - Reorders top candidates so the most contextually relevant evidence appears at rank 1, dramatically improving LLM generation fidelity and eliminating hallucinated source references.
4. **Stage 4 (Grounded Generation & Citation Verification — Google Gemini):**
   - Synthesizes grounded natural language answers restricted strictly to supplied sources with deterministic citation markers (`[1]`, `[2]`).

---

## Getting Started

### Prerequisites

* **Python:** `3.11.9`
* **Node.js:** `24.21.0` (or >= 18.x)
* **npm:** `11.19.0` (or >= 9.x)

---

### Backend Setup & Execution

1. **Create Virtual Environment & Install Dependencies:**
   From the repository root:
   ```powershell
   python -m venv backend\.venv
   backend\.venv\Scripts\python.exe -m pip install --upgrade pip
   backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
   ```

2. **Configure Environment Variables:**
   ```powershell
   Copy-Item .env.example backend\.env
   ```
   Add your Google Gemini API key to `backend/.env`:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   GEMINI_MODEL=gemini-3.8-flash
   ```
   *(To obtain a key: visit [Google AI Studio](https://aistudio.google.com/), generate an API key, and paste it into `backend/.env`).*

3. **Start the FastAPI Backend Server:**
   From the repository root:
   ```powershell
   $env:PYTHONPATH="backend"
   backend\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *(Note: Invoking `python.exe -m uvicorn` avoids Windows Application Control / execution policy restrictions on `.exe` stubs).*

   * Interactive Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   * Live Health Check: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
   * Search Endpoint: [http://127.0.0.1:8000/api/v1/retrieval/search](http://127.0.0.1:8000/api/v1/retrieval/search)
   * Grounded Q&A Endpoint: [http://127.0.0.1:8000/api/v1/qa/ask](http://127.0.0.1:8000/api/v1/qa/ask)
   * Evaluation Endpoint: [http://127.0.0.1:8000/api/v1/evaluation/results](http://127.0.0.1:8000/api/v1/evaluation/results)

4. **Run the Automated Test Suites:**
   * **Phase 2A (Ingestion & Chunking — 8 Test Suites):**
     ```powershell
     $env:PYTHONPATH="backend"
     backend\.venv\Scripts\python.exe backend\tests\test_phase2a_ingestion.py
     ```
   * **Phase 2B (Dense Vector & BM25 Indexing — 12 Test Suites):**
     ```powershell
     $env:PYTHONPATH="backend"
     backend\.venv\Scripts\python.exe backend\tests\test_phase2b_indexing.py
     ```
   * **Phase 3 (Hybrid Retrieval & RRF Fusion — 11 Test Suites):**
     ```powershell
     $env:PYTHONPATH="backend"
     backend\.venv\Scripts\python.exe backend\tests\test_phase3_retrieval.py
     ```
   * **Phase 4 (BGE Cross-Encoder Reranking — 12 Test Suites):**
     ```powershell
     $env:PYTHONPATH="backend"
     backend\.venv\Scripts\python.exe backend\tests\test_phase4_reranking.py
     ```
   * **Phase 5 (Gemini Generation & Citations — 15 Test Suites):**
     ```powershell
     $env:PYTHONPATH="backend"
     backend\.venv\Scripts\python.exe backend\tests\test_phase5_generation.py
     ```
   * **Phase 6 (Evaluation & Benchmarking — 16 Test Suites):**
     ```powershell
     $env:PYTHONPATH="backend"
     backend\.venv\Scripts\python.exe backend\tests\test_phase6_evaluation.py
     ```
   * **Run All 74 Automated Tests Sequentially:**
     ```powershell
     $env:PYTHONPATH="backend"
     backend\.venv\Scripts\python.exe backend\tests\test_phase2a_ingestion.py
     backend\.venv\Scripts\python.exe backend\tests\test_phase2b_indexing.py
     backend\.venv\Scripts\python.exe backend\tests\test_phase3_retrieval.py
     backend\.venv\Scripts\python.exe backend\tests\test_phase4_reranking.py
     backend\.venv\Scripts\python.exe backend\tests\test_phase5_generation.py
     backend\.venv\Scripts\python.exe backend\tests\test_phase6_evaluation.py
     ```

---

## Phase 6: RAG Evaluation, Comparison and Performance Benchmarking

Phase 6 provides a reproducible evaluation framework to measure retrieval quality, cross-encoder reranking effectiveness, answer generation quality, citation faithfulness, and stage-by-stage latencies.

### 1. Six Evaluated Retrieval Configurations
All experiments evaluate identical questions against the same corpus with standardized candidate pools:
1. **BM25 Lexical (No Rerank):** Pure inverted index lexical retrieval using BM25Okapi ($k_1=1.5, b=0.75$).
2. **Dense Vector (No Rerank):** Pure semantic dense vector retrieval via `all-MiniLM-L6-v2` embeddings and ChromaDB cosine distance.
3. **Hybrid RRF (No Rerank):** Multi-channel Reciprocal Rank Fusion ($k=60$) combining Dense and BM25 candidate lists.
4. **Hybrid RRF + BGE Rerank:** Hybrid RRF candidate generation followed by `BAAI/bge-reranker-base` cross-encoder sequence scoring.
5. **Dense Vector + BGE Rerank:** Dense vector candidate retrieval followed by BGE cross-encoder reranking.
6. **BM25 Lexical + BGE Rerank:** Sparse lexical candidate retrieval followed by BGE cross-encoder reranking.

### 2. Information Retrieval Metric Formulations

* **Precision@K:** Fraction of the top $K$ retrieved chunks that are ground-truth relevant:
  $$\text{Precision@K} = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$
  *(Note: Standard TREC/MS-MARCO convention: denominator remains $K$ even if fewer than $K$ results are returned, penalizing short retrieval lists).*

* **Recall@K:** Fraction of all known relevant chunks found in the top $K$:
  $$\text{Recall@K} = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$

* **Mean Reciprocal Rank (MRR):** Reciprocal position of the first relevant result averaged across queries:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_1(q)}$$
  *(Returns 0.0 if no relevant chunk appears in the retrieved candidates).*

* **Normalized Discounted Cumulative Gain (nDCG@K):** Evaluates ranked positions with graded relevance judgments:
  $$\text{DCG@K} = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
  Where $\text{rel}_i \in \{0, 1, 2\}$ and $\text{IDCG@K}$ is the ideal DCG attained by sorting all relevant chunks descending by relevance.

* **Hit Rate@K:** Binary indicator (1.0 if $|\text{Retrieved}_K \cap \text{Relevant}| > 0$, else 0.0).

### 3. Evaluation Dataset Taxonomy & Schema
The evaluation dataset is stored under `data/evaluation/dataset.json` with the following schema:
- `question_id`: Unique identifier (e.g. `eval_q001`).
- `question`: Query string.
- `expected_answer`: Human-verified ground-truth answer.
- `relevant_document_ids` / `relevant_chunk_ids`: Explicit supporting chunks.
- `relevance_judgments`: Graded mapping (e.g. `{"chunk_id": 2.0}`).
- `question_category`: Taxonomy category (`direct_factual`, `keyword_based`, `semantic`, `multi_passage`, `different_pages`, `insufficient_context`).
- `is_verified`: Boolean distinguishing human-curated ground truth from unverified candidate items.

### 4. Running Benchmarks
* **Via Standalone CLI Script:**
  ```powershell
  backend\.venv\Scripts\python.exe backend\scripts\run_evaluation.py --k-values 1 3 5 8
  ```
* **Via REST API:**
  - `POST /api/v1/evaluation/run`: Execute multi-configuration evaluation and return full report.
  - `GET /api/v1/evaluation/results`: Retrieve the most recent evaluation report.
  - `GET /api/v1/evaluation/dataset`: Inspect ground-truth questions and judgments.
  - `POST /api/v1/evaluation/dataset/add`: Extend dataset with new verified or candidate questions.
* **Via Interactive React Dashboard:**
  Navigate to the **RAG Evaluation & Benchmarking** section on the dashboard to inspect metric tables, toggle $K \in \{1, 3, 5, 8\}$, inspect per-query chunk rankings, and review latency profiles.

### 5. Empirical Benchmark Results Summary
Evaluated across 11 human-verified queries on the indexed technical corpus:

| Configuration | MRR | nDCG@3 | P@3 | R@3 | Hit@3 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 Lexical (No Rerank)** | 1.0000 | 0.9266 | 0.4074 | 0.8704 | 1.0000 | ~0.2 ms |
| **Dense Vector (No Rerank)** | 0.9444 | 0.9073 | 0.4074 | 0.8704 | 1.0000 | ~400 ms |
| **Hybrid RRF (No Rerank)** | 1.0000 | 0.9324 | 0.4074 | 0.8704 | 1.0000 | ~16 ms |
| **Hybrid RRF + BGE Rerank** | **1.0000** | **0.9776** | **0.4815** | **0.9630** | **1.0000** | ~1800 ms (CPU) |
| **Dense Vector + BGE Rerank** | 1.0000 | 0.9776 | 0.4815 | 0.9630 | 1.0000 | ~1580 ms (CPU) |
| **BM25 Lexical + BGE Rerank** | 1.0000 | 0.9776 | 0.4815 | 0.9630 | 1.0000 | ~1570 ms (CPU) |

**Key Finding:** Cross-encoder reranking produces measurable quality improvements across all candidate generators, increasing nDCG@3 from 0.9324 to **0.9776** and Recall@3 from 0.8704 to **0.9630**.

### 6. Answer Quality & Citation Faithfulness Rubric
- **Faithfulness Score:** 100% across verified ground-truth questions.
- **Citation Precision & Recall:** 100% (every cited reference substantiated claims; zero hallucinated citation IDs).
- **Negative Control Refusal:** 100% of out-of-domain queries correctly triggered refusal notices without hallucinating.

---

### Frontend Setup & Execution

1. **Navigate to the frontend directory:**
   ```powershell
   cd frontend
   ```

2. **Install Node.js Dependencies:**
   ```powershell
   npm install
   ```

3. **Start the Vite development server:**
   ```powershell
   npm run dev
   ```

4. **Production Build (Optional Verification):**
   ```powershell
   npm run build
   ```

5. **Open the Dashboard:**
   Visit [http://localhost:5173](http://localhost:5173). The dashboard automatically connects to the backend API, allowing you to drag and drop PDF files, inspect chunk text and metadata, monitor index health, compare BM25 vs. Dense vs. Hybrid RRF retrieval results, ask questions grounded by Google Gemini, and click citations to highlight source passages.

---

## Phase Roadmap

- [x] **Phase 1: Project Foundation**
  - Python 3.11.9 isolated virtual environment.
  - Modular FastAPI backend with `core/`, `models/`, `services/`, and `api/`.
  - Configured CORS middleware for the Vite development server.
  - React.js + Vite dashboard with live health check connectivity.
- [x] **Phase 2A: PDF Ingestion & Text Chunking**
  - PyMuPDF multi-page text extraction preserving 1-indexed page boundaries.
  - Whitespace and paragraph boundary cleaning.
  - Sentence-aware subword BPE chunking (~500 tokens, 75 overlap).
  - SHA-256 duplicate protection and persistent JSON catalog storage.
  - REST endpoints for upload, listing, chunk inspection, and deletion.
  - React DocumentManager with drag-and-drop, chunk inspection modal, and delete confirmation.
- [x] **Phase 2B: Dual Vector & Lexical Indexing**
  - ChromaDB persistent vector database setup with `sentence-transformers/all-MiniLM-L6-v2` embeddings (384-dim, CPU).
  - BM25Okapi sparse lexical token indexer with safe JSON persistence (no unsafe pickles).
  - Index orchestration service handling auto-indexing on upload, cascading deletion, and full index rebuilds.
  - REST API endpoints for `/api/v1/index/status`, `/api/v1/index/stats`, and `/api/v1/index/rebuild`.
  - Frontend IndexManager component with live ChromaDB metrics, BM25 corpus stats, synchronization status, and rebuild trigger.
  - 12 comprehensive automated test suites covering embeddings, ChromaDB, BM25, synchronization, and benchmarks.
- [x] **Phase 3: Hybrid Retrieval Using Reciprocal Rank Fusion (RRF)**
  - Standalone Dense semantic retrieval using cosine distance on ChromaDB.
  - Standalone BM25 lexical keyword retrieval with tokenization and IDF smoothing.
  - Reciprocal Rank Fusion (RRF) algorithm: $\text{RRF}(d) = \sum \frac{1}{k + \text{rank}_m(d)}$ with configurable $k=60$.
  - Modular `RetrievalService` supporting `'dense'`, `'bm25'`, and `'hybrid'`.
  - REST API endpoint: `POST /api/v1/retrieval/search` returning ranked results, citations, scores, ranks, and isolated timings.
  - React `RetrievalComparison` component with search query input, strategy toggle, top K tuning, score badges, and citation tags.
  - 11 automated test suites covering dense search, BM25 search, mathematical RRF formula verification, rank fusion, deduplication, missing result handling, deterministic sorting, and latency isolation.
- [x] **Phase 4: Cross-Encoder Reranking (`BAAI/bge-reranker-base`)**
  - Modular `RerankerService` singleton with lazy loading, CPU execution, and PyTorch `inference_mode()`.
  - Sigmoid calibration mapping raw classification logits into normalized relevance scores $[0, 1]$.
  - Micro-batch processing (`batch_size=16`) and truncation (`max_length=512`) protecting 12 GB RAM budget.
  - Unified retrieval integration in `retrieval_service.py` preserving `original_rank`, `rerank_rank`, `reranker_score`, and source citations.
  - Extended API request schema (`rerank`, `reranker_top_k`, `reranker_batch_size`) and response schema (`rerank_enabled`, `reranking_time_ms`, `reranker_model`).
  - Enhanced React UI with cross-encoder toggle button, parameter tuning panel, rerank rank badges with position movement indicators (`▲ +1`, `▼ -1`, `=`), and isolated rerank latency display.
  - 12 automated unit and integration test suites covering singleton lifecycle, real PyTorch CPU forward pass, query-chunk pair formatting, descending score ordering, citation preservation, empty query resilience, batching, truncation, and API controller validation.
- [x] **Phase 5: LLM Generation & Interactive Citations UI (`google-genai`)**
  - Modular `GenerationService` utilizing official `google-genai` Python SDK with `genai.Client`.
  - Configurable Gemini model (`GEMINI_MODEL`, default `gemini-3.8-flash`), temperature, and timeout.
  - Deterministic 1-based citation mapping (`[1]`, `[2]`, ...) preserving physical page numbers and distinct chunk IDs.
  - Strict anti-hallucination grounding prompt: instruction injection defense, source boundary enforcement, and insufficient-information detection.
  - Dedicated React `QuestionAnswer` component with clickable citation pills.
  - 15 automated test suites passing 100%.
- [x] **Phase 6: RAG Evaluation, Comparison and Performance Benchmarking**
  - Six standardized retrieval configurations: BM25, Dense, Hybrid RRF, Hybrid RRF + BGE Rerank, Dense + BGE Rerank, BM25 + BGE Rerank.
  - Pure mathematical metric functions: Precision@K, Recall@K, MRR, nDCG@K (logarithmic discount), HitRate@K.
  - Human-verified ground-truth benchmark dataset across 6 taxonomy categories with explicit unverified candidate tagging.
  - Transparent answer faithfulness and citation attribution rubric with automated negative control refusal checks.
  - Stage-by-stage latency distribution benchmarking (Embedding, BM25, ChromaDB, RRF, BGE, Gemini) and RSS process memory tracking.
  - Reproducible CLI runner (`backend/scripts/run_evaluation.py`) and REST endpoints (`POST /api/v1/evaluation/run`, `GET /api/v1/evaluation/results`, `GET /api/v1/evaluation/dataset`).
  - Interactive React `EvaluationDashboard` with strategy comparison matrix, metric cutoff tuning ($K \in \{1, 3, 5, 8\}$), per-query ground-truth inspection, and latency distribution charts.
  - 16 comprehensive automated test modules passing with 100% pass rate.

---

## Limitations of Context-Based RAG

1. **Context Boundary:** The LLM only sees the top $K$ retrieved and reranked passages, not entire multi-hundred page documents. If a query requires aggregating disparate facts separated by hundreds of un-retrieved pages, answer synthesis may report insufficient context.
2. **Document Formatting:** Highly complex nested multi-column tables and scanned image PDFs require OCR preprocessing before text chunking.
3. **API Rate Limits:** Free-tier Gemini API keys may be subject to requests-per-minute (RPM) limits. The system handles HTTP 429 errors gracefully with informative user alerts.

---

## Troubleshooting

- **Error: `Google Gemini API key is not configured`**
  - Ensure you copied `.env.example` to `backend/.env` and supplied a valid `GEMINI_API_KEY`.
- **Error: `Application Control policy has blocked pip.exe`**
  - Run Python modules using `python -m <module>` rather than executing `.exe` stubs directly.
- **Port 8000 or 5173 already in use:**
  - Change `PORT` in `backend/.env` or specify `npm run dev -- --port 5174`.
