"""
Cricket World Cup RAG — Embedding Generator
============================================
Production-grade embedding generation using sentence-transformers/all-MiniLM-L6-v2.
Free, local model with no API costs. Handles batching and normalization.

Vector dimension: 384
Similarity metric: Cosine (via normalized inner product)
"""

import logging
from typing import List

import numpy as np

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class EmbeddingGenerator:
    """
    Generates normalized embeddings using sentence-transformers/all-MiniLM-L6-v2.

    Features:
        - Local model, no API costs
        - Automatic batching for efficiency
        - L2 normalization for cosine similarity via inner product
        - Input validation
        - Deterministic output (same input → same embedding)
    """

    def __init__(self):
        self._model_name = EMBEDDING_MODEL
        self._dimension = EMBEDDING_DIMENSION
        self._batch_size = EMBEDDING_BATCH_SIZE
        self._model = None

    @property
    def model(self):
        """Lazy-initialize sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self._model_name}")
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required. Install with: pip install sentence-transformers"
                )
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine similarity via inner product."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return vectors / norms

    def _validate_input(self, text: str) -> str:
        """Validate and sanitize input text."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        text = text.strip()
        if not text:
            raise ValueError("Empty text cannot be embedded")
        # Replace null bytes
        text = text.replace("\x00", " ")
        return text

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate a normalized embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            np.ndarray of shape (384,), L2-normalized.
        """
        text = self._validate_input(text)
        embeddings = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embeddings[0].astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate normalized embeddings for a batch of texts.
        Automatically processes in batches for efficiency.

        Args:
            texts: List of input texts.

        Returns:
            np.ndarray of shape (len(texts), 384), L2-normalized.
        """
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        validated = [self._validate_input(t) for t in texts]

        logger.info(f"Embedding {len(validated)} texts in batches of {self._batch_size}...")

        # sentence-transformers handles batching internally, but we'll batch for logging
        all_embeddings = []
        for i in range(0, len(validated), self._batch_size):
            batch = validated[i : i + self._batch_size]
            batch_num = i // self._batch_size + 1
            total_batches = (len(validated) + self._batch_size - 1) // self._batch_size

            logger.debug(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)...")

            embeddings = self.model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            all_embeddings.append(embeddings)

        vectors = np.vstack(all_embeddings).astype(np.float32)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query. Alias for embed_single with query semantics.

        Args:
            query: Search query text.

        Returns:
            np.ndarray of shape (384,), L2-normalized.
        """
        return self.embed_single(query)
