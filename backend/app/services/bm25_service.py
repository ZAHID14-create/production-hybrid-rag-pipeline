import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.models.document import ChunkMetadata

logger = logging.getLogger(__name__)


class BM25IndexService:
    """Persistent BM25 lexical indexing service.

    Avoids unsafe pickle loading by persisting raw corpus and chunk metadata in JSON,
    and rebuilding the fast in-memory BM25Okapi inverted index during initialization or updates.
    """

    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = Path(index_dir or settings.BM25_INDEX_DIRECTORY)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_file = self.index_dir / "corpus.json"

        # In-memory index structures
        self.bm25: Optional[BM25Okapi] = None
        # chunk_id -> dict with chunk metadata and tokens
        self.chunk_store: Dict[str, Dict[str, Any]] = {}
        # Ordered list of chunk_ids aligned with BM25 doc index
        self.chunk_ids: List[str] = []

        self._load_and_build_index()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into normalized lowercase alphanumeric tokens for BM25.

        Strips punctuation and accents while retaining alphanumeric terms, acronyms,
        and hyphenated/compound technical terms.
        """
        if not text:
            return []
        # Match sequences of alphanumeric characters and words
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
        # Filter out purely punctuation artifacts if any
        return [t.strip(".-_") for t in tokens if len(t.strip(".-_")) > 0]

    def _build_bm25_model(self, tokenized_corpus: List[List[str]]) -> Optional[BM25Okapi]:
        """Instantiates BM25Okapi model and applies positive IDF floor for small corpora."""
        if not tokenized_corpus:
            return None
        model = BM25Okapi(tokenized_corpus)
        # Apply positive IDF floor for small corpora or terms appearing in >= 50% of docs
        for word, idf_val in model.idf.items():
            if idf_val <= 0.0:
                model.idf[word] = 0.25
        return model

    def _load_and_build_index(self) -> None:
        """Loads corpus from JSON and builds the in-memory BM25Okapi index."""
        if not self.corpus_file.exists():
            self.bm25 = None
            self.chunk_store = {}
            self.chunk_ids = []
            return

        try:
            with open(self.corpus_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.chunk_store = data.get("chunks", {})
            self.chunk_ids = list(self.chunk_store.keys())

            if self.chunk_ids:
                tokenized_corpus = [
                    self.chunk_store[cid]["tokens"]
                    for cid in self.chunk_ids
                ]
                self.bm25 = self._build_bm25_model(tokenized_corpus)
                logger.info("BM25 index built with %d chunks from %s", len(self.chunk_ids), self.corpus_file)
            else:
                self.bm25 = None
        except Exception as e:
            logger.error("Failed to load BM25 corpus from %s: %s", self.corpus_file, e)
            self.bm25 = None
            self.chunk_store = {}
            self.chunk_ids = []

    def _save_corpus(self) -> None:
        """Persists the chunk corpus to disk safely via temporary file write."""
        temp_file = self.corpus_file.with_suffix(".tmp")
        payload = {
            "version": "1.0",
            "total_chunks": len(self.chunk_ids),
            "chunks": self.chunk_store,
        }
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        temp_file.replace(self.corpus_file)

    def add_chunks(self, chunks: List[ChunkMetadata]) -> int:
        """Adds or updates chunks in the BM25 index and updates the persistent corpus.

        Uses stable chunk_ids to prevent duplicates.
        Returns the number of new/updated chunks.
        """
        if not chunks:
            return 0

        for chunk in chunks:
            tokens = self.tokenize(chunk.chunk_text)
            self.chunk_store[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
                "token_count": chunk.token_count,
                "character_count": chunk.character_count,
                "tokens": tokens,
            }

        self.chunk_ids = list(self.chunk_store.keys())
        tokenized_corpus = [self.chunk_store[cid]["tokens"] for cid in self.chunk_ids]
        self.bm25 = self._build_bm25_model(tokenized_corpus)

        self._save_corpus()
        logger.info("BM25 index updated: %d total chunks", len(self.chunk_ids))
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        """Removes all chunks associated with a document_id and rebuilds the BM25 index."""
        initial_count = len(self.chunk_store)
        to_delete = [
            cid for cid, meta in self.chunk_store.items()
            if meta.get("document_id") == document_id
        ]

        if not to_delete:
            return 0

        for cid in to_delete:
            del self.chunk_store[cid]

        self.chunk_ids = list(self.chunk_store.keys())

        if self.chunk_ids:
            tokenized_corpus = [self.chunk_store[cid]["tokens"] for cid in self.chunk_ids]
            self.bm25 = self._build_bm25_model(tokenized_corpus)
        else:
            self.bm25 = None

        self._save_corpus()
        deleted_count = initial_count - len(self.chunk_store)
        logger.info("Deleted %d BM25 chunks for document %s", deleted_count, document_id)
        return deleted_count

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs lexical BM25 search for the given query text.

        Returns ranked list of dicts containing score, chunk_id, document_id,
        document_name, page_number, chunk_index, and chunk_text.
        """
        if not self.bm25 or not self.chunk_ids or not query.strip():
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Pair scores with chunk_ids
        scored_pairs: List[Tuple[str, float]] = []
        for idx, score in enumerate(scores):
            if score > 0.0:  # Only return non-zero relevance scores
                scored_pairs.append((self.chunk_ids[idx], float(score)))

        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = scored_pairs[:top_k]

        results = []
        for rank, (cid, score) in enumerate(top_pairs, start=1):
            meta = self.chunk_store[cid]
            results.append({
                "rank": rank,
                "score": score,
                "chunk_id": meta["chunk_id"],
                "document_id": meta["document_id"],
                "document_name": meta["document_name"],
                "page_number": meta["page_number"],
                "chunk_index": meta["chunk_index"],
                "chunk_text": meta["chunk_text"],
                "token_count": meta.get("token_count", 0),
            })

        return results

    def reset_index(self) -> None:
        """Clears all in-memory structures and deletes the corpus file."""
        self.chunk_store = {}
        self.chunk_ids = []
        self.bm25 = None
        if self.corpus_file.exists():
            self.corpus_file.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """Returns diagnostic statistics for the BM25 index."""
        corpus_size = self.corpus_file.stat().st_size if self.corpus_file.exists() else 0
        return {
            "status": "ready" if self.bm25 and len(self.chunk_ids) > 0 else "empty",
            "chunk_count": len(self.chunk_ids),
            "corpus_file": str(self.corpus_file),
            "corpus_size_bytes": corpus_size,
        }


bm25_service = BM25IndexService()
