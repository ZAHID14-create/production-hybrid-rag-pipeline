from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.api.routes_health import router as health_router
from app.api.routes_documents import router as documents_router
from app.api.routes_index import router as index_router
from app.api.routes_retrieval import router as retrieval_router
from app.api.routes_qa import router as qa_router
from app.api.routes_evaluation import router as evaluation_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production Hybrid RAG Pipeline combining BM25 Keyword Search, Dense Vector Search (ChromaDB), BGE Cross-Encoder Reranking, and Google Gemini with Source Citations.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS Middleware for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers under /api/v1
app.include_router(health_router, prefix=settings.API_V1_PREFIX)
app.include_router(documents_router, prefix=settings.API_V1_PREFIX)
app.include_router(index_router, prefix=settings.API_V1_PREFIX)
app.include_router(retrieval_router, prefix=settings.API_V1_PREFIX)
app.include_router(qa_router, prefix=settings.API_V1_PREFIX)
app.include_router(evaluation_router, prefix=settings.API_V1_PREFIX)


@app.on_event("startup")
async def startup_event():
    """Validates and warms up the vector store collection on server boot."""
    try:
        from app.services.vector_store import vector_store
        col = vector_store.collection
        count = col.count()
        print(f"[STARTUP] ChromaDB collection '{col.name}' connected (UUID: {col.id}, chunks: {count}).")
    except Exception as e:
        print(f"[STARTUP WARNING] Failed initial ChromaDB check: {e}")


@app.get("/", tags=["Root"])
async def root():
    """Root info endpoint providing service links and metadata."""
    return JSONResponse(
        content={
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs": "/docs",
            "health": f"{settings.API_V1_PREFIX}/health",
            "documents": f"{settings.API_V1_PREFIX}/documents",
            "index": f"{settings.API_V1_PREFIX}/index/status",
            "retrieval": f"{settings.API_V1_PREFIX}/retrieval/search",
            "qa": f"{settings.API_V1_PREFIX}/qa/ask",
            "evaluation": f"{settings.API_V1_PREFIX}/evaluation/results",
            "phase": "Phase 6 - RAG Evaluation, Comparison and Performance Benchmarking",
        }
    )
