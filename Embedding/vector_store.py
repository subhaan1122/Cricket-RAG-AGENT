"""
Cricket World Cup RAG — FAISS Vector Store + BM25 Hybrid Search
=================================================================
Production-grade vector store with FAISS HNSWFlat for semantic search,
BM25 for keyword search, and hybrid score fusion.

Index: HNSWFlat (HNSW graph with flat storage)
Distance: Inner Product on L2-normalized vectors (= Cosine similarity)
Dimension: 384
Hybrid: BM25 keyword search + FAISS semantic search with configurable weights
"""

import json
import logging
import math
import os
import pickle
import re
import shutil
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    EMBEDDING_DIMENSION,
    FAISS_INDEX_PATH,
    FAISS_HNSW_M,
    FAISS_EF_CONSTRUCTION,
    FAISS_EF_SEARCH,
    CHUNKS_JSON_PATH,
    VECTORS_JSON_PATH,
    BM25_INDEX_PATH,
    INDEX_DIR,
    BM25_ENABLED,
    BM25_K1,
    BM25_B,
    BM25_WEIGHT,
    SEMANTIC_WEIGHT,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


# ────────────────────────────────────────────────────────────
# BM25 INDEX
# ────────────────────────────────────────────────────────────

class BM25Index:
    """
    In-memory BM25 index for keyword-based search.
    Complements FAISS semantic search for hybrid retrieval.
    """

    # Cricket-specific stop words to exclude from tokenization
    STOP_WORDS = frozenset({
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "were", "are", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "it", "its", "this",
        "that", "these", "those", "i", "you", "he", "she", "we", "they", "me",
        "him", "her", "us", "them", "my", "your", "his", "our", "their", "what",
        "which", "who", "whom", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "not", "only", "same", "so", "than", "too", "very", "just", "because",
        "as", "if", "then", "also", "about", "up", "out", "into", "over",
        "after", "before", "between", "under", "again", "further", "once",
    })

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: Dict[str, int] = {}  # chunk_id → token count
        self.doc_tokens: Dict[str, List[str]] = {}  # chunk_id → tokens
        self.df: Counter = Counter()  # term → document frequency
        self.chunk_ids: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25, keeping cricket-relevant terms."""
        text = text.lower()
        # Keep numbers (important for cricket stats: years, scores, averages)
        tokens = re.findall(r'[a-z0-9]+(?:\'[a-z]+)?', text)
        # Remove stop words but keep short cricket terms
        return [t for t in tokens if t not in self.STOP_WORDS or t.isdigit()]

    def add_document(self, chunk_id: str, text: str) -> None:
        """Add a document to the BM25 index."""
        tokens = self._tokenize(text)
        self.doc_tokens[chunk_id] = tokens
        self.doc_lengths[chunk_id] = len(tokens)

        # Update document frequency (unique terms in this doc)
        unique_terms = set(tokens)
        for term in unique_terms:
            self.df[term] += 1

        if chunk_id not in self.chunk_ids:
            self.chunk_ids.append(chunk_id)

        self.doc_count = len(self.chunk_ids)
        if self.doc_count > 0:
            self.avg_doc_len = sum(self.doc_lengths.values()) / self.doc_count

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search using BM25 scoring.

        Returns:
            List of (chunk_id, bm25_score) sorted by descending score.
        """
        if self.doc_count == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self.df:
                continue

            idf = math.log((self.doc_count - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1.0)

            for chunk_id in self.chunk_ids:
                tokens = self.doc_tokens.get(chunk_id, [])
                tf = tokens.count(term)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths.get(chunk_id, 1)
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
                )
                scores[chunk_id] += idf * tf_norm

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save(self, path: Path) -> None:
        """Persist BM25 index to disk."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "doc_tokens": self.doc_tokens,
            "df": dict(self.df),
            "chunk_ids": self.chunk_ids,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved BM25 index: {self.doc_count} documents → {path}")

    def load(self, path: Path) -> bool:
        """Load BM25 index from disk. Returns True on success."""
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.k1 = data["k1"]
            self.b = data["b"]
            self.doc_count = data["doc_count"]
            self.avg_doc_len = data["avg_doc_len"]
            self.doc_lengths = data["doc_lengths"]
            self.doc_tokens = data["doc_tokens"]
            self.df = Counter(data["df"])
            self.chunk_ids = data["chunk_ids"]
            logger.info(f"Loaded BM25 index: {self.doc_count} documents from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def reset(self) -> None:
        """Clear all BM25 data."""
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.doc_lengths.clear()
        self.doc_tokens.clear()
        self.df.clear()
        self.chunk_ids.clear()


class FAISSVectorStore:
    """
    FAISS-backed vector store with HNSWFlat index + BM25 hybrid search.

    Features:
        - HNSWFlat index for fast approximate nearest neighbor search
        - BM25 keyword search for hybrid retrieval
        - Hybrid score fusion (weighted combination of semantic + keyword)
        - Cosine similarity via inner product on normalized vectors
        - Incremental insert (add vectors without rebuilding)
        - Atomic disk persistence with fail-safe writes
        - Thread-safe operations
        - Vector ID ↔ chunk ID mapping
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        dimension: int = EMBEDDING_DIMENSION,
        hnsw_m: int = FAISS_HNSW_M,
        ef_construction: int = FAISS_EF_CONSTRUCTION,
        ef_search: int = FAISS_EF_SEARCH,
    ):
        self._index_dir = Path(index_dir) if index_dir else INDEX_DIR
        self._dimension = dimension
        self._hnsw_m = hnsw_m
        self._ef_construction = ef_construction
        self._ef_search = ef_search

        self._index = None
        self._vector_to_chunk: Dict[int, str] = {}  # vector_id → chunk_id
        self._chunk_to_vector: Dict[str, int] = {}   # chunk_id → vector_id
        self._lock = threading.Lock()
        self._next_id = 0

        # BM25 hybrid search
        self._bm25: Optional[BM25Index] = None
        self._bm25_enabled = BM25_ENABLED

        # Ensure index directory exists
        self._index_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self._index_path = self._index_dir / "faiss.index"
        self._chunks_path = self._index_dir / "chunks.json"
        self._vectors_path = self._index_dir / "vectors.json"
        self._bm25_path = self._index_dir / "bm25.pkl"

    @property
    def faiss(self):
        """Lazy import of faiss."""
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu package required. Install with: pip install faiss-cpu"
            )

    @property
    def is_initialized(self) -> bool:
        return self._index is not None

    @property
    def total_vectors(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        return self._dimension

    def initialize(self, force_new: bool = False) -> None:
        """
        Initialize the FAISS index and BM25 index. Load from disk if exists,
        otherwise create new indices.

        Args:
            force_new: If True, discard existing index and create fresh.
        """
        with self._lock:
            if not force_new and self._index_path.exists():
                self._load_from_disk()
            else:
                self._create_new_index()

            # Initialize BM25
            if self._bm25_enabled:
                self._bm25 = BM25Index()
                if not force_new and self._bm25_path.exists():
                    self._bm25.load(self._bm25_path)
                else:
                    logger.info("Created new BM25 index")

    def _create_new_index(self) -> None:
        """Create a fresh HNSWFlat index."""
        faiss = self.faiss

        # HNSWFlat with inner product (cosine on normalized vectors)
        self._index = faiss.IndexHNSWFlat(self._dimension, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self._ef_construction
        self._index.hnsw.efSearch = self._ef_search

        self._vector_to_chunk = {}
        self._chunk_to_vector = {}
        self._next_id = 0

        logger.info(
            f"Created new FAISS HNSWFlat index: "
            f"dim={self._dimension}, M={self._hnsw_m}, "
            f"efConstruction={self._ef_construction}, efSearch={self._ef_search}"
        )

    def _load_from_disk(self) -> None:
        """Load FAISS index and mappings from disk."""
        faiss = self.faiss

        try:
            self._index = faiss.read_index(str(self._index_path))
            self._index.hnsw.efSearch = self._ef_search

            # Load vector ↔ chunk mappings
            if self._vectors_path.exists():
                with open(self._vectors_path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                self._vector_to_chunk = {int(k): v for k, v in mapping.get("vector_to_chunk", {}).items()}
                self._chunk_to_vector = mapping.get("chunk_to_vector", {})
                self._next_id = max(self._vector_to_chunk.keys(), default=-1) + 1
            else:
                self._vector_to_chunk = {}
                self._chunk_to_vector = {}
                self._next_id = self._index.ntotal

            logger.info(
                f"Loaded FAISS index from disk: "
                f"{self._index.ntotal} vectors, "
                f"{len(self._vector_to_chunk)} mappings"
            )
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}. Creating new index.")
            self._create_new_index()

    def add_vectors(
        self,
        vectors: np.ndarray,
        chunk_ids: List[str],
        chunk_texts: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Add vectors to the FAISS index with chunk ID mapping.
        Optionally also index texts in BM25.

        Args:
            vectors: np.ndarray of shape (N, 384), L2-normalized.
            chunk_ids: List of chunk IDs corresponding to each vector.
            chunk_texts: Optional list of texts for BM25 indexing.

        Returns:
            List of assigned vector IDs.
        """
        if vectors.shape[0] != len(chunk_ids):
            raise ValueError(
                f"Vector count ({vectors.shape[0]}) != chunk_id count ({len(chunk_ids)})"
            )
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Vector dimension ({vectors.shape[1]}) != expected ({self._dimension})"
            )

        if not self.is_initialized:
            self.initialize()

        with self._lock:
            vectors = vectors.astype(np.float32)
            assigned_ids = []

            for i in range(vectors.shape[0]):
                chunk_id = chunk_ids[i]

                # Skip if chunk already indexed
                if chunk_id in self._chunk_to_vector:
                    assigned_ids.append(self._chunk_to_vector[chunk_id])
                    continue

                vector_id = self._next_id
                self._next_id += 1

                # Add to FAISS (HNSWFlat uses sequential add)
                self._index.add(vectors[i : i + 1])

                # Update mappings
                self._vector_to_chunk[vector_id] = chunk_id
                self._chunk_to_vector[chunk_id] = vector_id
                assigned_ids.append(vector_id)

            logger.info(f"Added {len(assigned_ids)} vectors to FAISS index")

            # Add to BM25 index if texts are provided
            if self._bm25 and chunk_texts:
                for i, chunk_id in enumerate(chunk_ids):
                    if i < len(chunk_texts) and chunk_id not in self._bm25.doc_tokens:
                        self._bm25.add_document(chunk_id, chunk_texts[i])

            return assigned_ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[int, str, float]]:
        """
        Search for nearest neighbors in the FAISS index.

        Args:
            query_vector: np.ndarray of shape (384,), L2-normalized.
            top_k: Number of results to return.

        Returns:
            List of (vector_id, chunk_id, similarity_score) tuples,
            sorted by descending similarity.
        """
        if not self.is_initialized:
            self.initialize()

        if self._index.ntotal == 0:
            return []

        with self._lock:
            query = query_vector.reshape(1, -1).astype(np.float32)
            effective_k = min(top_k, self._index.ntotal)

            distances, indices = self._index.search(query, effective_k)

            results = []
            for i in range(effective_k):
                idx = int(indices[0][i])
                score = float(distances[0][i])

                if idx < 0:  # FAISS returns -1 for not-found
                    continue

                chunk_id = self._vector_to_chunk.get(idx, f"unknown_{idx}")
                results.append((idx, chunk_id, score))

            return results

    def hybrid_search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 20,
        semantic_weight: float = SEMANTIC_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
    ) -> List[Tuple[int, str, float]]:
        """
        Hybrid search combining FAISS semantic search + BM25 keyword search.
        Uses Reciprocal Rank Fusion (RRF) for score combination.

        Args:
            query_vector: np.ndarray of shape (384,), L2-normalized.
            query_text: Raw query text for BM25 search.
            top_k: Number of results to return.
            semantic_weight: Weight for FAISS semantic scores.
            bm25_weight: Weight for BM25 keyword scores.

        Returns:
            List of (vector_id, chunk_id, hybrid_score) tuples,
            sorted by descending hybrid score.
        """
        if not self._bm25_enabled or not self._bm25 or self._bm25.doc_count == 0:
            return self.search(query_vector, top_k)

        # Get candidates from both sources (fetch more to fuse)
        candidate_k = min(top_k * 3, self._index.ntotal) if self._index else top_k * 3
        semantic_results = self.search(query_vector, top_k=candidate_k)
        bm25_results = self._bm25.search(query_text, top_k=candidate_k)

        if not semantic_results and not bm25_results:
            return []

        # Normalize semantic scores to [0, 1]
        if semantic_results:
            max_sem = max(r[2] for r in semantic_results)
            min_sem = min(r[2] for r in semantic_results)
            sem_range = max_sem - min_sem if max_sem > min_sem else 1.0
        else:
            sem_range = 1.0
            min_sem = 0.0

        # Normalize BM25 scores to [0, 1]
        if bm25_results:
            max_bm25 = max(r[1] for r in bm25_results)
            min_bm25 = min(r[1] for r in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
        else:
            bm25_range = 1.0
            min_bm25 = 0.0

        # Build score maps
        semantic_scores: Dict[str, Tuple[int, float]] = {}
        for vid, cid, score in semantic_results:
            norm_score = (score - min_sem) / sem_range if sem_range > 0 else 0.5
            semantic_scores[cid] = (vid, norm_score)

        bm25_scores: Dict[str, float] = {}
        for cid, score in bm25_results:
            norm_score = (score - min_bm25) / bm25_range if bm25_range > 0 else 0.5
            bm25_scores[cid] = norm_score

        # Fuse scores
        all_chunk_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())
        fused = []

        for cid in all_chunk_ids:
            sem_data = semantic_scores.get(cid)
            sem_score = sem_data[1] if sem_data else 0.0
            vid = sem_data[0] if sem_data else -1

            bm25_score = bm25_scores.get(cid, 0.0)

            # If only in BM25 results, we need the vector_id
            if vid == -1:
                vid = self._chunk_to_vector.get(cid, -1)

            hybrid_score = (semantic_weight * sem_score) + (bm25_weight * bm25_score)
            fused.append((vid, cid, hybrid_score))

        # Sort by hybrid score
        fused.sort(key=lambda x: x[2], reverse=True)
        return fused[:top_k]

    def add_text_to_bm25(self, chunk_id: str, text: str) -> None:
        """Add a single document to the BM25 index."""
        if self._bm25 and chunk_id not in self._bm25.doc_tokens:
            self._bm25.add_document(chunk_id, text)

    def save(self) -> None:
        """
        Persist FAISS index and mappings to disk.
        Uses atomic writes with temp files to prevent corruption.
        """
        if not self.is_initialized:
            logger.warning("No index to save")
            return

        with self._lock:
            self._index_dir.mkdir(parents=True, exist_ok=True)

            # Atomic write: write to temp, then rename
            tmp_index = self._index_path.with_suffix(".index.tmp")
            tmp_vectors = self._vectors_path.with_suffix(".json.tmp")

            try:
                # Save FAISS index
                self.faiss.write_index(self._index, str(tmp_index))

                # Save mappings
                mapping = {
                    "vector_to_chunk": {str(k): v for k, v in self._vector_to_chunk.items()},
                    "chunk_to_vector": self._chunk_to_vector,
                    "next_id": self._next_id,
                    "total_vectors": self._index.ntotal,
                    "dimension": self._dimension,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(tmp_vectors, "w", encoding="utf-8") as f:
                    json.dump(mapping, f, indent=2)

                # Atomic rename
                if tmp_index.exists():
                    shutil.move(str(tmp_index), str(self._index_path))
                if tmp_vectors.exists():
                    shutil.move(str(tmp_vectors), str(self._vectors_path))

                logger.info(
                    f"Saved FAISS index: {self._index.ntotal} vectors → {self._index_path}"
                )

                # Save BM25 index
                if self._bm25 and self._bm25.doc_count > 0:
                    self._bm25.save(self._bm25_path)

            except Exception as e:
                # Cleanup temp files on failure
                for tmp in [tmp_index, tmp_vectors]:
                    if tmp.exists():
                        tmp.unlink()
                logger.error(f"Failed to save FAISS index: {e}")
                raise

    def get_chunk_id(self, vector_id: int) -> Optional[str]:
        """Get chunk ID for a vector ID."""
        return self._vector_to_chunk.get(vector_id)

    def has_chunk(self, chunk_id: str) -> bool:
        """Check if a chunk ID is already indexed."""
        return chunk_id in self._chunk_to_vector

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_vectors": self.total_vectors,
            "dimension": self._dimension,
            "index_type": "HNSWFlat",
            "hnsw_m": self._hnsw_m,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
            "metric": "cosine",
            "mappings_count": len(self._vector_to_chunk),
            "index_file_exists": self._index_path.exists(),
            "bm25_enabled": self._bm25_enabled,
            "bm25_documents": self._bm25.doc_count if self._bm25 else 0,
        }

    def reset(self) -> None:
        """Delete the index and all mappings. Destructive."""
        with self._lock:
            self._create_new_index()
            # Reset BM25
            if self._bm25:
                self._bm25.reset()
            # Remove files
            for path in [self._index_path, self._vectors_path, self._bm25_path]:
                if path.exists():
                    path.unlink()
            logger.warning("FAISS + BM25 index reset — all vectors deleted")
