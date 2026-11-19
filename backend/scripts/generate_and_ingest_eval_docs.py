import sys
from pathlib import Path
from io import BytesIO
import pymupdf
from fastapi import UploadFile

# Ensure backend package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.routes_documents import upload_document

def create_and_ingest():
    doc = pymupdf.open()
    pages = [
        ("Dense Vector Embeddings and ChromaDB Semantic Search",
         "Dense semantic retrieval in our production hybrid RAG pipeline utilizes the sentence-transformers/all-MiniLM-L6-v2 embedding model. This pre-trained transformer maps arbitrary text sequences into a 384-dimensional dense vector space. In this vector space, semantic similarity correlates directly with cosine proximity. The local vector store is managed by ChromaDB using persistent disk storage under the data/chroma_db directory. ChromaDB constructs an HNSW (Hierarchical Navigable Small World) graph with cosine distance metric to achieve approximate nearest neighbor lookups in logarithmic time. The default HNSW construction parameters use M=16 bidirectional links per node and efConstruction=100 search depth during construction. Vector queries compute normalized cosine distance and return candidate chunks sorted by similarity."),
        ("BM25 Lexical Keyword Matching and Inverted Indexing",
         "Sparse lexical retrieval is powered by the BM25Okapi algorithm, which builds an inverted index over the tokenized corpus. Each document chunk is tokenized into lowercase alphanumeric terms with punctuation stripped. The BM25 scoring function balances term frequency (TF) and inverse document frequency (IDF) while correcting for document length variations. The parameter k1 governs term frequency saturation and is set to the standard value of 1.5, preventing single repeated keywords from dominating relevance scores. The parameter b controls document length penalization and is configured to 0.75, which normalizes long chunks against the average corpus chunk length. The BM25 index stores raw chunk metadata in JSON format to guarantee deterministic and safe index reconstruction without relying on insecure Python pickling."),
        ("Reciprocal Rank Fusion (RRF) Hybrid Integration",
         "Reciprocal Rank Fusion (RRF) is an unsupervised rank aggregation method that combines ranked candidate lists from disparate retrieval systems into a unified candidate pool. In our pipeline, RRF integrates dense semantic candidates from ChromaDB with sparse lexical candidates from BM25. The mathematical formula for scoring chunk d across retrieval systems m is: RRF(d) = sum(1 / (k + rank_m(d))), where rank_m(d) is the 1-based rank position of chunk d in system m, and k is a smoothing constant set to 60. RRF eliminates the need to normalize raw cosine distances and BM25 scores onto a shared numerical scale, avoiding calibration distortion caused by unbounded score distributions. Chunks appearing in top ranks across both dense and sparse retrieval achieve the highest combined RRF scores."),
        ("Cross-Encoder Precision Reranking with BGE",
         "While bi-encoder embeddings provide fast sub-linear search over large collections, they process queries and documents independently, missing fine-grained token-level cross-interactions. To overcome this limitation, our pipeline introduces a second-stage cross-encoder reranker powered by BAAI/bge-reranker-base. The cross-encoder receives the concatenated query and candidate text tokens separated by special boundary tokens. It performs full all-to-all self-attention across every query token and document token simultaneously. The model classification head outputs a raw logit that is passed through a sigmoid function to produce a calibrated relevance probability bounded between 0.0 and 1.0. Reranking top-10 hybrid candidates on CPU requires approximately 45 to 65 milliseconds, providing high precision re-ordering."),
        ("Grounded Generation and Anti-Hallucination Guardrails",
         "The generative answering stage employs Google Gemini (configured with gemini-3.8-flash) to synthesize factual answers grounded strictly in the retrieved context. The LLM prompt enforces strict anti-hallucination guardrails: the model must rely solely on the facts presented in the numbered context passages. When attributing factual assertions, the model must append verbatim citation tokens such as [1] or [2] referencing the corresponding passage identifier. If the provided context passages do not contain sufficient evidence to answer the query, the model is explicitly instructed to refuse to answer and declare that the documents do not contain sufficient information. The decoding temperature is set to a deterministic 0.2 with a maximum output limit of 1024 tokens."),
        ("Information Retrieval Evaluation Metrics and Safeguards",
         "Information retrieval evaluation quantifies retrieval efficacy across six standardized configurations using macro-averaged metrics. Precision@K measures the fraction of the top K retrieved chunks that are relevant, defined as relevant retrieved divided by K. Recall@K measures the proportion of all known relevant chunks that appear in the top K results. Mean Reciprocal Rank (MRR) evaluates the rank of the first relevant result, averaging 1 / rank_1 across queries. Normalized Discounted Cumulative Gain (nDCG@K) accounts for graded relevance positions using a logarithmic discount factor log2(rank + 1). Hit Rate@K indicates whether at least one relevant document was retrieved in the top K. Evaluating systems with identical ground-truth judgments prevents algorithmic bias and ensures reproducible benchmarking.")
    ]

    for title, content in pages:
        p = doc.new_page(width=595, height=842)
        p.insert_textbox(pymupdf.Rect(50, 50, 545, 90), f"{title}\n", fontsize=16)
        p.insert_textbox(pymupdf.Rect(50, 100, 545, 750), content, fontsize=12)

    pdf_bytes = doc.tobytes()
    doc_name = "hybrid_rag_technical_specification.pdf"
    
    # Save into documents directory
    docs_dir = Path(__file__).resolve().parent.parent.parent / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = docs_dir / doc_name
    pdf_path.write_bytes(pdf_bytes)
    print(f"Written PDF to {pdf_path} ({len(pdf_bytes)} bytes)")

    # Upload using existing application endpoint logic
    upload_file = UploadFile(filename=doc_name, file=BytesIO(pdf_bytes))
    res = upload_document(file=upload_file)
    print(f"Ingested doc_id={res.document.document_id}, chunks={res.chunks_created}")

if __name__ == "__main__":
    create_and_ingest()
