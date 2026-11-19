import logging
from typing import List, Optional
import torch
from transformers import AutoModel, AutoTokenizer

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """CPU-optimized dense vector embedding service using Hugging Face Transformers.

    Utilizes sentence-transformers/all-MiniLM-L6-v2 architecture with mean pooling
    and L2 normalization. Bypasses scikit-learn dependency blocks by operating directly
    via PyTorch and HuggingFace AutoModel/AutoTokenizer.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = torch.device("cpu")
        self.dimension = 384  # all-MiniLM-L6-v2 hidden size

        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None

    def _ensure_loaded(self) -> None:
        """Lazy loads the tokenizer and model into CPU memory."""
        if self._tokenizer is None or self._model is None:
            logger.info("Loading embedding model '%s' on %s...", self.model_name, self.device)
            # Try loading cached files first without network requests
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self._model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)

            self._model.to(self.device)
            self._model.eval()
            logger.info("Embedding model loaded successfully.")

    @staticmethod
    def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """Computes mean pooling of token embeddings weighted by attention mask."""
        token_embeddings = model_output[0]  # First element has last hidden state: [batch_size, seq_len, hidden_dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generates normalized dense vector embeddings for a list of texts.

        Args:
            texts: List of input text strings to embed.
            batch_size: Batch size for CPU tensor inference.

        Returns:
            List of 384-dimensional unit-norm float embedding vectors.
        """
        if not texts:
            return []

        self._ensure_loaded()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                model_output = self._model(**encoded)
                sentence_embeddings = self._mean_pooling(model_output, encoded["attention_mask"])
                # L2 normalize embeddings so cosine distance equals dot product
                normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            all_embeddings.extend(normalized_embeddings.cpu().tolist())

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generates embedding for a single query string."""
        if not query.strip():
            return [0.0] * self.dimension
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else [0.0] * self.dimension


embedding_service = EmbeddingService()
