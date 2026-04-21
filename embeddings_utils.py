"""
Cricket World Cup RAG — Embeddings & Search Utilities
======================================================
Manages the FAISS + BM25 hybrid index, generates embeddings,
performs hybrid search with re-ranking, and assembles
optimized context for LLM prompt construction.

Features:
    - Hybrid search (FAISS semantic + BM25 keyword)
    - Metadata-aware re-ranking
    - Multi-query search with deduplication
    - Context assembly with intelligent truncation
    - Index build/rebuild management
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Embedding.embeddings import EmbeddingGenerator
from Embedding.vector_store import FAISSVectorStore
from Embedding.chunking import TextChunker, Chunk
from Embedding.ingestion import IngestionPipeline

from config import (
    INDEX_DIR,
    CRICKET_EMBEDDINGS_DIR,
    CRICKET_METADATA_DIR,
    CHUNKS_JSON_PATH,
    SEARCH_TOP_K_DEFAULT,
    SEARCH_SCORE_THRESHOLD,
    BM25_ENABLED,
    BM25_WEIGHT,
    SEMANTIC_WEIGHT,
    RERANK_ENABLED,
    RERANK_TOP_N,
    RERANK_METADATA_BOOST,
    CROSS_ENCODER_RERANK,
    CROSS_ENCODER_MODEL,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNKS,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# World Cup years for metadata matching
WORLD_CUP_YEARS = {"2003", "2007", "2011", "2015", "2019", "2023"}


class EmbeddingsManager:
    """
    Manages the FAISS + BM25 hybrid index, embeddings, and search
    for the Cricket World Cup RAG chatbot.

    Responsibilities:
        - Load / initialize FAISS + BM25 indices + chunk mappings
        - Embed user queries
        - Perform hybrid search → re-rank → return ranked snippets
        - Multi-query search with deduplication and diversity
        - Context assembly optimized for LLM consumption
        - Rebuild index from Cricket Data if needed
        - Provide index statistics
    """

    def __init__(self, index_dir: Optional[Path] = None):
        self._index_dir = Path(index_dir) if index_dir else INDEX_DIR
        self._embedder: Optional[EmbeddingGenerator] = None
        self._store: Optional[FAISSVectorStore] = None
        self._pipeline: Optional[IngestionPipeline] = None
        self._chunker: Optional[TextChunker] = None
        self._cross_encoder = None
        self._initialized = False

    # ────────────────────────────────────────────────────────
    # INITIALIZATION
    # ────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Load the embedding model, FAISS + BM25 indices, and chunk mappings.
        If no index exists on disk, creates fresh ones.
        """
        if self._initialized:
            return

        logger.info("Initializing EmbeddingsManager...")

        self._index_dir.mkdir(parents=True, exist_ok=True)

        # Core components (reuse existing modules)
        self._embedder = EmbeddingGenerator()
        self._store = FAISSVectorStore(index_dir=self._index_dir)
        self._store.initialize()
        self._chunker = TextChunker()
        self._pipeline = IngestionPipeline(
            embedding_generator=self._embedder,
            vector_store=self._store,
            chunker=self._chunker,
        )

        # Load cross-encoder for precise re-ranking
        if CROSS_ENCODER_RERANK:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
                logger.info(f"Cross-encoder loaded: {CROSS_ENCODER_MODEL}")
            except Exception as e:
                logger.warning(f"Cross-encoder unavailable, falling back to metadata rerank: {e}")
                self._cross_encoder = None

        self._initialized = True

        # Populate BM25 from existing chunks if BM25 is empty but chunks exist
        self._ensure_bm25_populated()

        stats = self.get_stats()
        logger.info(
            f"EmbeddingsManager ready — "
            f"{stats['total_vectors']} vectors, "
            f"{stats['total_chunks']} chunks, "
            f"BM25: {stats.get('bm25_documents', 0)} docs"
        )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "EmbeddingsManager not initialized. Call initialize() first."
            )

    def _ensure_bm25_populated(self) -> None:
        """Populate BM25 index from existing chunks if it's empty."""
        if (
            self._store
            and self._store._bm25
            and self._store._bm25.doc_count == 0
            and self._pipeline
            and self._pipeline.total_chunks > 0
        ):
            logger.info(
                f"BM25 index is empty — populating from {self._pipeline.total_chunks} existing chunks..."
            )
            all_chunks = self._pipeline.get_all_chunks()
            count = 0
            for chunk_id, chunk in all_chunks.items():
                self._store.add_text_to_bm25(chunk_id, chunk.text)
                count += 1
            # Persist the rebuilt BM25 index
            if count > 0:
                self._store._bm25.save(self._store._bm25_path)
                logger.info(f"BM25 index populated and saved: {count} documents")

    # ────────────────────────────────────────────────────────
    # HYBRID SEARCH (FAISS + BM25)
    # ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = SEARCH_TOP_K_DEFAULT,
        score_threshold: float = SEARCH_SCORE_THRESHOLD,
        bm25_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search over the FAISS + BM25 indices.

        Args:
            query: User question / search text.
            top_k: Maximum results to return.
            score_threshold: Minimum score to include.
            bm25_weight: Override BM25 weight (default from config).

        Returns:
            List of dicts with keys:
                text, file_name, score, chunk_type, section_title, tags, chunk_id
        """
        self._ensure_initialized()

        if self._store.total_vectors == 0:
            logger.warning("Index is empty — no vectors to search")
            return []

        # Embed query
        query_vector = self._embedder.embed_query(query)

        # Perform hybrid search
        effective_bm25_weight = bm25_weight if bm25_weight is not None else BM25_WEIGHT
        effective_semantic_weight = 1.0 - effective_bm25_weight

        if BM25_ENABLED and self._store._bm25 and self._store._bm25.doc_count > 0:
            raw_results = self._store.hybrid_search(
                query_vector, query,
                top_k=top_k * 2,  # Get more candidates for re-ranking
                semantic_weight=effective_semantic_weight,
                bm25_weight=effective_bm25_weight,
            )
        else:
            raw_results = self._store.search(query_vector, top_k=top_k * 2)

        # Resolve chunks and filter by score
        results = []
        for vector_id, chunk_id, score in raw_results:
            if score < score_threshold:
                continue

            chunk = self._pipeline.get_chunk(chunk_id)
            if not chunk:
                continue

            results.append({
                "text": chunk.text,
                "file_name": chunk.metadata.file_name,
                "score": round(float(score), 4),
                "chunk_type": chunk.metadata.chunk_type,
                "section_title": chunk.metadata.section_title,
                "tags": chunk.metadata.tags,
                "chunk_id": chunk.chunk_id,
            })

        # Re-rank results using metadata
        if RERANK_ENABLED and results:
            results = self._rerank(results, query, top_k)

        results = results[:top_k]

        logger.info(
            f"Search for '{query[:60]}...' → "
            f"{len(results)} results (top_k={top_k}, threshold={score_threshold})"
        )
        return results

    def _rerank(
        self,
        results: List[Dict[str, Any]],
        query: str,
        top_n: int = RERANK_TOP_N,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using cross-encoder (if available) + metadata matching.
        """
        # Phase 1: Cross-encoder re-ranking (semantic precision)
        if self._cross_encoder and len(results) > 1:
            try:
                pairs = [(query, r["text"][:512]) for r in results[:top_n]]
                ce_scores = self._cross_encoder.predict(pairs)
                # Normalize cross-encoder scores to [0, 1]
                ce_min, ce_max = float(min(ce_scores)), float(max(ce_scores))
                ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
                for i, r in enumerate(results[:top_n]):
                    ce_norm = (float(ce_scores[i]) - ce_min) / ce_range
                    # Blend: 60% cross-encoder, 40% original hybrid score
                    r["score"] = round(0.6 * ce_norm + 0.4 * r["score"], 4)
                logger.info(f"Cross-encoder re-ranked {len(pairs)} candidates")
            except Exception as e:
                logger.warning(f"Cross-encoder re-rank failed, using metadata only: {e}")

        # Phase 2: Metadata boosting
        query_lower = query.lower()

        # Extract years from query
        query_years = set(re.findall(r'\b(2003|2007|2011|2015|2019|2023)\b', query))

        for r in results:
            boost = 0.0
            text_lower = r["text"].lower()
            fname_lower = r["file_name"].lower()
            tags = r.get("tags", [])

            # Year match boost
            if query_years:
                chunk_years = set(re.findall(r'\b(2003|2007|2011|2015|2019|2023)\b', r["text"]))
                tag_years = {t.split(":")[1] for t in tags if t.startswith("year:")}
                all_chunk_years = chunk_years | tag_years
                if query_years & all_chunk_years:
                    boost += RERANK_METADATA_BOOST.get("year_match", 0.0)

            # Memorable moments boost
            if "memorable_moments" in fname_lower or r["chunk_type"] == "memorable_moments":
                boost += RERANK_METADATA_BOOST.get("memorable_moments", 0.0)

            # Cross-tournament / records boost for cross-tournament queries
            cross_keywords = ["all", "every", "across", "history", "all-time", "record", "overall"]
            if any(kw in query_lower for kw in cross_keywords):
                if "cross_tournament" in fname_lower or r["chunk_type"] in ("cross_tournament", "records_and_facts"):
                    boost += RERANK_METADATA_BOOST.get("type_match", 0.0)

            # Player mention boost
            player_keywords = [
                "kohli", "tendulkar", "dhoni", "ponting", "rohit", "warner",
                "starc", "gayle", "stokes", "williamson", "head", "shami",
                "de villiers", "sangakkara", "gilchrist", "mcgrath", "malinga",
                "yuvraj", "bumrah", "guptill", "shakib", "sehwag",
            ]
            for player in player_keywords:
                if player in query_lower and player in text_lower:
                    boost += RERANK_METADATA_BOOST.get("player_match", 0.0)
                    break

            # Team mention boost
            team_keywords = [
                "india", "australia", "england", "new zealand", "pakistan",
                "south africa", "sri lanka", "bangladesh", "west indies",
                "afghanistan", "ireland",
            ]
            for team in team_keywords:
                if team in query_lower and team in text_lower:
                    boost += RERANK_METADATA_BOOST.get("team_match", 0.0)
                    break

            r["score"] = round(r["score"] + boost, 4)

        # Re-sort by boosted score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def multi_search(
        self,
        queries: List[str],
        top_k: int = SEARCH_TOP_K_DEFAULT,
        score_threshold: float = SEARCH_SCORE_THRESHOLD,
        bm25_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform multiple searches and merge results, deduplicating by chunk_id.
        Ensures diversity by including top results from each query.

        Args:
            queries: List of search queries.
            top_k: Max results per query.
            score_threshold: Minimum similarity.
            bm25_weight: Override BM25 weight.

        Returns:
            Merged, deduplicated list of results sorted by score.
        """
        self._ensure_initialized()
        seen_chunks = set()
        all_results = []

        for q in queries:
            results = self.search(
                q, top_k=top_k,
                score_threshold=score_threshold,
                bm25_weight=bm25_weight,
            )
            for r in results:
                if r["chunk_id"] not in seen_chunks:
                    seen_chunks.add(r["chunk_id"])
                    all_results.append(r)

        # Sort by score descending
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results

    def get_context_text(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = SEARCH_SCORE_THRESHOLD,
        bm25_weight: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get formatted context text for RAG prompt construction.

        Args:
            query: User question.
            top_k: Number of chunks to retrieve.
            score_threshold: Minimum similarity.
            bm25_weight: Override BM25 weight.

        Returns:
            (context_text, sources) — ready for LLM prompt injection.
        """
        results = self.search(
            query, top_k=top_k,
            score_threshold=score_threshold,
            bm25_weight=bm25_weight,
        )

        if not results:
            return "", []

        context_parts = []
        sources = []

        for r in results:
            context_parts.append(
                f"[Source: {r['file_name']} | Type: {r['chunk_type']} | Score: {r['score']}]\n{r['text']}"
            )
            sources.append({
                "file": r["file_name"],
                "score": r["score"],
                "type": r["chunk_type"],
                "title": r["section_title"],
            })

        context_text = "\n\n---\n\n".join(context_parts)
        return context_text, sources

    def multi_query_context(
        self,
        queries: List[str],
        top_k: int = 10,
        score_threshold: float = SEARCH_SCORE_THRESHOLD,
        max_total: int = MAX_CONTEXT_CHUNKS,
        bm25_weight: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get context from multiple search queries, merged and deduplicated.
        Ensures diverse coverage by including results from each query.
        Applies smart truncation to stay within context limits.

        Args:
            queries: List of search queries.
            top_k: Max results per query.
            score_threshold: Minimum similarity.
            max_total: Maximum total results to include.
            bm25_weight: Override BM25 weight.

        Returns:
            (context_text, sources)
        """
        results = self.multi_search(
            queries, top_k=top_k,
            score_threshold=score_threshold,
            bm25_weight=bm25_weight,
        )
        results = results[:max_total]

        if not results:
            return "", []

        context_parts = []
        sources = []
        total_chars = 0

        for r in results:
            part = f"[Source: {r['file_name']} | Type: {r['chunk_type']} | Score: {r['score']}]\n{r['text']}"

            # Smart truncation — stop if we'd exceed context limit
            if total_chars + len(part) + 10 > MAX_CONTEXT_CHARS:
                remaining = len(results) - len(context_parts)
                if remaining > 0:
                    logger.info(f"Context truncated: included {len(context_parts)} of {len(results)} chunks ({total_chars} chars)")
                break

            context_parts.append(part)
            total_chars += len(part) + 10

            sources.append({
                "file": r["file_name"],
                "score": r["score"],
                "type": r["chunk_type"],
                "title": r["section_title"],
            })

        context_text = "\n\n---\n\n".join(context_parts)
        return context_text, sources

    # ────────────────────────────────────────────────────────
    # INDEX MANAGEMENT
    # ────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> Dict[str, int]:
        """
        Build (or rebuild) the FAISS + BM25 index from the Cricket Data embeddings.

        If index already exists and force_rebuild=False, only new
        (non-duplicate) content is added incrementally.

        Args:
            force_rebuild: If True, wipe and rebuild from scratch.

        Returns:
            Ingestion statistics dict.
        """
        self._ensure_initialized()

        if force_rebuild:
            logger.warning("Force rebuild requested — resetting index")
            self._store.reset()
            # Also clear chunks.json so dedup starts fresh
            chunks_path = Path(INDEX_DIR) / "chunks.json"
            if chunks_path.exists():
                chunks_path.unlink()
                logger.info("Cleared chunks.json for full rebuild")
            # Clear BM25
            bm25_path = Path(INDEX_DIR) / "bm25.pkl"
            if bm25_path.exists():
                bm25_path.unlink()
                logger.info("Cleared bm25.pkl for full rebuild")
            self._pipeline = IngestionPipeline(
                embedding_generator=self._embedder,
                vector_store=self._store,
                chunker=self._chunker,
            )

        logger.info("Building index from Cricket World Cup dataset...")
        stats = self._pipeline.ingest_cricket_dataset(
            tags=["cricket", "world_cup"]
        )

        logger.info(
            f"Index build complete — "
            f"{stats.get('chunks_created', 0)} new chunks, "
            f"{stats.get('vectors_added', 0)} new vectors"
        )
        return stats

    def add_document(
        self,
        file_path: str,
        source_type: str = "document",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Incrementally add a single document to the index.
        Useful for adding new World Cup data without full rebuild.

        Args:
            file_path: Path to the document file.
            source_type: Type of source document.
            tags: Optional tags.

        Returns:
            Ingestion stats.
        """
        self._ensure_initialized()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        tags = tags or ["cricket", "world_cup"]
        result = self._pipeline.ingest_file(
            file_path=path,
            source_type=source_type,
            tags=tags,
        )

        # Persist
        self._store.save()
        self._pipeline._save_chunks()
        logger.info(f"Added '{path.name}' → {result.get('vectors_added', 0)} vectors")
        return result

    # ────────────────────────────────────────────────────────
    # UTILITIES
    # ────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self._initialized:
            return {"initialized": False}

        store_stats = self._store.get_stats() if self._store else {}

        return {
            "initialized": True,
            "total_vectors": self._store.total_vectors if self._store else 0,
            "total_chunks": self._pipeline.total_chunks if self._pipeline else 0,
            "index_dir": str(self._index_dir),
            "index_file_exists": (self._index_dir / "faiss.index").exists(),
            "chunks_file_exists": (self._index_dir / "chunks.json").exists(),
            "bm25_file_exists": (self._index_dir / "bm25.pkl").exists(),
            "bm25_documents": store_stats.get("bm25_documents", 0),
            "embedding_model": self._embedder._model_name if self._embedder else None,
            "embedding_dimension": self._embedder.dimension if self._embedder else None,
            "hybrid_search": BM25_ENABLED,
        }

    @property
    def is_ready(self) -> bool:
        """Check if index is initialized and has vectors."""
        return self._initialized and self._store is not None and self._store.total_vectors > 0

    @property
    def total_vectors(self) -> int:
        return self._store.total_vectors if self._store else 0

    @property
    def total_chunks(self) -> int:
        return self._pipeline.total_chunks if self._pipeline else 0
