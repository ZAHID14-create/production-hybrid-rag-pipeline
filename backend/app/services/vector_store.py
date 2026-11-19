import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.models.document import ChunkMetadata
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_chunks"


class ChromaVectorStore:
    """Persistent ChromaDB vector store for dense embedding indexing and similarity search."""

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = Path(persist_directory or settings.CHROMA_PERSIST_DIRECTORY)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    def _get_client(self) -> chromadb.PersistentClient:
        """Returns or initializes the singleton PersistentClient with standard persistent settings."""
        if self._client is None:
            logger.info("Initializing persistent ChromaDB client at '%s'...", self.persist_directory)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
            )
        return self._client

    def _get_or_refresh_collection(self, force_refresh: bool = False):
        """
        Retrieves the collection by configured name ('rag_chunks').
        Guarantees self-healing by detecting stale collection UUIDs and re-fetching
        the active collection by name from the database.
        """
        client = self._get_client()

        if self._collection is None or force_refresh:
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Connected to ChromaDB collection '%s' (UUID: %s, chunks: %d).",
                COLLECTION_NAME,
                self._collection.id,
                self._collection.count(),
            )
            return self._collection

        # Validate that the cached collection UUID exists in the underlying database
        try:
            _ = self._collection.count()
            return self._collection
        except Exception as e:
            err_str = str(e).lower()
            if "does not exist" in err_str or "not found" in err_str:
                logger.warning(
                    "Cached ChromaDB collection ID '%s' is stale/invalid (%s). Re-fetching '%s' by name...",
                    getattr(self._collection, "id", "unknown"),
                    e,
                    COLLECTION_NAME,
                )
                self._collection = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                return self._collection
            raise

    def _ensure_connected(self) -> None:
        """Backward-compatible helper ensuring client and collection are connected."""
        self._get_or_refresh_collection()

    @property
    def collection(self):
        """Active ChromaDB collection, guaranteed connected and validated."""
        return self._get_or_refresh_collection()

    def add_chunks(self, chunks: List[ChunkMetadata], batch_size: int = 64) -> int:
        """Generates embeddings and upserts chunks into ChromaDB.

        Uses stable chunk_ids to ensure idempotent operations.
        Returns the number of upserted chunks.
        """
        if not chunks:
            return 0

        col = self._get_or_refresh_collection()
        texts = [chunk.chunk_text for chunk in chunks]

        # Compute embeddings in CPU batches
        embeddings = embedding_service.embed_texts(texts, batch_size=batch_size)

        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "page_number": int(chunk.page_number),
                "chunk_index": int(chunk.chunk_index),
                "token_count": int(chunk.token_count),
                "character_count": int(chunk.character_count),
            }
            for chunk in chunks
        ]

        # Upsert into ChromaDB in batches
        for i in range(0, len(chunks), batch_size):
            end_idx = i + batch_size
            try:
                col.upsert(
                    ids=ids[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                )
            except Exception as e:
                if "does not exist" in str(e).lower():
                    logger.warning("Upsert hit stale collection reference. Reconnecting and retrying batch...")
                    col = self._get_or_refresh_collection(force_refresh=True)
                    col.upsert(
                        ids=ids[i:end_idx],
                        embeddings=embeddings[i:end_idx],
                        documents=texts[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                    )
                else:
                    raise

        logger.info(
            "Upserted %d chunks into ChromaDB. Total collection count: %d",
            len(chunks),
            col.count(),
        )
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        """Deletes all chunks associated with a document_id from ChromaDB."""
        col = self._get_or_refresh_collection()
        try:
            initial_count = col.count()
        except Exception:
            col = self._get_or_refresh_collection(force_refresh=True)
            initial_count = col.count()

        try:
            col.delete(where={"document_id": document_id})
        except Exception as e:
            if "does not exist" in str(e).lower():
                logger.warning("Delete hit stale collection reference. Reconnecting and retrying...")
                col = self._get_or_refresh_collection(force_refresh=True)
                col.delete(where={"document_id": document_id})
            else:
                logger.error("Error deleting document %s from ChromaDB: %s", document_id, e)
                return 0

        final_count = col.count()
        deleted_count = initial_count - final_count
        logger.info("Deleted %d chunks for document %s from ChromaDB", deleted_count, document_id)
        return deleted_count

    def search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Queries ChromaDB using a precomputed embedding vector.

        Returns ranked list of dicts with cosine similarity scores and chunk metadata.
        """
        col = self._get_or_refresh_collection()
        try:
            if col.count() == 0:
                return []
        except Exception:
            col = self._get_or_refresh_collection(force_refresh=True)
            if col.count() == 0:
                return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k, col.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        try:
            results = col.query(**kwargs)
        except Exception as e:
            if "does not exist" in str(e).lower():
                logger.warning("Search query hit stale collection reference. Reconnecting and retrying query...")
                col = self._get_or_refresh_collection(force_refresh=True)
                kwargs["n_results"] = min(top_k, col.count())
                results = col.query(**kwargs)
            else:
                raise

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        ranked_results = []
        for rank, (cid, doc_text, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            # In cosine space, cosine distance = 1 - cosine_similarity.
            # Thus cosine similarity = 1 - distance.
            similarity = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0

            ranked_results.append({
                "rank": rank,
                "score": round(similarity, 6),
                "distance": round(float(dist), 6) if dist is not None else 0.0,
                "chunk_id": cid,
                "document_id": meta.get("document_id", ""),
                "document_name": meta.get("document_name", ""),
                "page_number": int(meta.get("page_number", 1)),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "chunk_text": doc_text,
                "token_count": int(meta.get("token_count", 0)),
            })

        return ranked_results

    def search_by_text(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Embeds query text and retrieves most similar chunks."""
        query_vector = embedding_service.embed_query(query)
        return self.search_by_vector(query_vector, top_k=top_k)

    def reset_collection(self) -> None:
        """Deletes the current collection and re-creates an empty one."""
        client = self._get_client()
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' reset to empty state.", COLLECTION_NAME)

    def get_stats(self) -> Dict[str, Any]:
        """Returns diagnostic statistics for the ChromaDB vector store."""
        try:
            col = self._get_or_refresh_collection()
            count = col.count()
            status = "ready" if count > 0 else "empty"
        except Exception as e:
            logger.warning("Initial ChromaDB stats check failed (%s). Refreshing collection...", e)
            try:
                col = self._get_or_refresh_collection(force_refresh=True)
                count = col.count()
                status = "ready" if count > 0 else "empty"
            except Exception as e2:
                logger.error("Failed to query ChromaDB stats after refresh: %s", e2)
                status = "error"
                count = 0

        return {
            "status": status,
            "collection_name": COLLECTION_NAME,
            "chunk_count": count,
            "persist_directory": str(self.persist_directory),
            "embedding_dimension": embedding_service.dimension,
            "embedding_model_name": embedding_service.model_name,
        }

        return {
            "status": status,
            "collection_name": COLLECTION_NAME,
            "chunk_count": count,
            "persist_directory": str(self.persist_directory),
            "embedding_dimension": embedding_service.dimension,
            "embedding_model_name": embedding_service.model_name,
        }


vector_store = ChromaVectorStore()
