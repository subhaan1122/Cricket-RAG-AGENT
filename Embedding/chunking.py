"""
Cricket World Cup RAG — Text Chunking Engine
================================================
Production-grade chunking with semantic boundary detection,
cricket-specific type classification, rich metadata extraction,
and overlap support.

Chunk size: 150-500 tokens (target 300)
Overlap: 50 tokens
Preserves: headings, paragraphs, tables, speaker turns, cricket sections
"""

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import (
    CHUNK_SIZE_MIN,
    CHUNK_SIZE_MAX,
    CHUNK_SIZE_TARGET,
    CHUNK_OVERLAP,
    CHUNK_TYPES,
    SECTION_BREAK_PATTERNS,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# Pre-compile section break patterns
_SECTION_PATTERNS = [re.compile(p, re.MULTILINE) for p in SECTION_BREAK_PATTERNS]


@dataclass
class ChunkMetadata:
    """Canonical metadata schema for each chunk."""
    source_type: str          # document | meeting | image | csv | excel
    source_id: str            # Unique source identifier
    file_name: str            # Original filename
    chunk_id: str             # UUID for this chunk
    chunk_index: int          # Position within the source document
    chunk_type: str           # document_section | table | meeting_transcript | etc.
    section_title: Optional[str] = None
    language: str = "en"
    created_at: str = ""      # ISO-8601
    indexed_at: str = ""      # ISO-8601 (set at index time)
    version: int = 1
    hash: str = ""            # SHA-256 of chunk text
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "file_name": self.file_name,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
            "section_title": self.section_title,
            "language": self.language,
            "created_at": self.created_at,
            "indexed_at": self.indexed_at,
            "version": self.version,
            "hash": self.hash,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkMetadata":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Chunk:
    """A text chunk with metadata, ready for embedding."""
    text: str
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id

    @property
    def hash(self) -> str:
        return self.metadata.hash

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            text=d["text"],
            metadata=ChunkMetadata.from_dict(d["metadata"]),
        )


def compute_text_hash(text: str) -> str:
    """SHA-256 hash of normalized text for deduplication."""
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """
    Estimate token count. Approximation: 1 token ≈ 4 characters for English.
    Good enough for chunking boundaries; exact count is model-dependent.
    """
    return max(1, len(text) // 4)


def _find_section_boundaries(text: str) -> List[int]:
    """
    Find line indices that represent semantic section boundaries.
    Returns sorted list of line indices where sections start.
    """
    lines = text.split("\n")
    boundaries = set()
    boundaries.add(0)  # Start is always a boundary

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in _SECTION_PATTERNS:
            if pattern.search(stripped):
                boundaries.add(i)
                break

    return sorted(boundaries)


def _detect_chunk_type(text: str, file_name: str, section_title: Optional[str]) -> str:
    """Classify chunk type based on content and source."""
    text_lower = text.lower()
    fname_lower = file_name.lower()

    # Cricket-specific types — memorable moments and rich context
    if "memorable_moments" in fname_lower or "memorable moments" in text_lower[:200]:
        return "memorable_moments"
    if "cross_tournament" in fname_lower or "all-time records" in text_lower[:200]:
        return "cross_tournament"
    if "records_and_facts" in fname_lower:
        return "records_and_facts"
    if "questions_answers" in fname_lower or "cricket_questions" in fname_lower:
        return "qa_pair"

    # Cricket-specific types — tournament & match data
    if "tournament summary" in text_lower or "tournament_summary" in fname_lower:
        return "tournament_summary"
    if "icc cricket world cup" in text_lower and ("team standings" in text_lower or "team performance" in text_lower):
        return "tournament_summary"
    if fname_lower.startswith("all_player_statistics") or (
        "player:" in text_lower and "world cups:" in text_lower
    ):
        return "player_statistics"
    if fname_lower == "player_stats.json" or ("player:" in text_lower and "batting:" in text_lower):
        return "player_statistics"
    if fname_lower == "match_index.json" or text_lower.startswith("icc cricket world cup — match index"):
        return "match_index"
    if "world_cup_summary" in fname_lower and ("captain performance" in text_lower or "head-to-head" in text_lower or "team performance" in text_lower):
        return "tournament_summary"
    if any(
        marker in text_lower
        for marker in ["match summary:", "batting highlights:", "bowling highlights:"]
    ):
        return "match_embedding"
    if "statistical analysis:" in text_lower and "innings analysis" in text_lower:
        return "match_embedding"

    # Meeting transcript detection
    if re.search(r"^\[\d{2}:\d{2}", text, re.MULTILINE):
        return "meeting_transcript"
    if re.search(r"^(Speaker|Participant)\s*\d*\s*:", text, re.MULTILINE):
        return "meeting_transcript"

    # Table detection
    if text.count("|") > 4 and text.count("\n") > 2:
        return "table"

    # File-extension-based fallbacks
    if fname_lower.endswith(".csv"):
        return "csv_group"
    if fname_lower.endswith((".xlsx", ".xls")):
        return "excel_sheet"

    # Summary detection
    if section_title and "summary" in section_title.lower():
        return "summary"

    return "document_section"


def _extract_section_title(text: str) -> Optional[str]:
    """Extract section title from the beginning of a chunk."""
    first_line = text.strip().split("\n")[0].strip()

    # Markdown heading
    heading_match = re.match(r"^#{1,6}\s+(.+)$", first_line)
    if heading_match:
        return heading_match.group(1).strip()

    # Cricket embedding section headers
    for pattern in [
        "Match Summary:", "Batting Highlights:", "Bowling Highlights:",
        "Captain Performance:", "Key Moments:", "Result:", "Team Standings:",
        "Captains:", "ICC Cricket World Cup",
    ]:
        if first_line.startswith(pattern):
            return first_line.rstrip(":")

    # Short title-like first line
    if len(first_line) < 80 and not first_line.endswith("."):
        return first_line

    return None


class TextChunker:
    """
    Semantic-aware text chunker.

    Strategy:
        1. Split text at semantic boundaries (headings, sections)
        2. Within each section, split at paragraph boundaries
        3. Merge small fragments up to target size
        4. Add overlap between consecutive chunks
        5. Classify each chunk by type
    """

    def __init__(
        self,
        chunk_size_min: int = CHUNK_SIZE_MIN,
        chunk_size_max: int = CHUNK_SIZE_MAX,
        chunk_size_target: int = CHUNK_SIZE_TARGET,
        overlap: int = CHUNK_OVERLAP,
    ):
        self.min_tokens = chunk_size_min
        self.max_tokens = chunk_size_max
        self.target_tokens = chunk_size_target
        self.overlap = overlap

    def chunk_text(
        self,
        text: str,
        source_type: str = "document",
        source_id: str = "",
        file_name: str = "",
        tags: Optional[List[str]] = None,
        created_at: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Chunk a document into semantically coherent pieces with metadata.

        Args:
            text: Full document text.
            source_type: One of document|meeting|image|csv|excel.
            source_id: Unique identifier for the source document.
            file_name: Original filename.
            tags: Optional tags for all chunks from this document.
            created_at: ISO-8601 creation timestamp.

        Returns:
            List of Chunk objects, each with full metadata.
        """
        if not text or not text.strip():
            return []

        tags = tags or []
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        source_id = source_id or compute_text_hash(text)[:16]

        # Step 1: Split into semantic sections
        sections = self._split_into_sections(text)

        # Step 2: Split large sections into sub-chunks
        raw_chunks = []
        for section in sections:
            if estimate_tokens(section) <= self.max_tokens:
                raw_chunks.append(section)
            else:
                raw_chunks.extend(self._split_large_section(section))

        # Step 3: Merge tiny fragments
        merged = self._merge_small_chunks(raw_chunks)

        # Step 4: Add overlap
        overlapped = self._add_overlap(merged)

        # Step 5: Create Chunk objects with metadata
        chunks = []
        for idx, chunk_text in enumerate(overlapped):
            if not chunk_text.strip():
                continue

            section_title = _extract_section_title(chunk_text)
            chunk_type = _detect_chunk_type(chunk_text, file_name, section_title)
            text_hash = compute_text_hash(chunk_text)

            metadata = ChunkMetadata(
                source_type=source_type,
                source_id=source_id,
                file_name=file_name,
                chunk_id=str(uuid.uuid4()),
                chunk_index=idx,
                chunk_type=chunk_type,
                section_title=section_title,
                language="en",
                created_at=created_at,
                indexed_at="",  # Set at index time
                version=1,
                hash=text_hash,
                tags=tags.copy(),
            )
            chunks.append(Chunk(text=chunk_text.strip(), metadata=metadata))

        logger.info(
            f"Chunked '{file_name}' → {len(chunks)} chunks "
            f"(source_type={source_type})"
        )
        return chunks

    def chunk_meeting_transcript(
        self,
        transcript: dict,
        source_id: str = "",
        file_name: str = "conversation.json",
        tags: Optional[List[str]] = None,
        created_at: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Chunk a meeting transcript (conversation.json).
        Each speaker turn becomes one chunk.

        Args:
            transcript: Parsed JSON with speaker turns.
                Expected format: {"turns": [{"speaker": ..., "text": ..., "timestamp": ...}]}
                or list of turn dicts directly.
            source_id: Unique meeting identifier.
            file_name: Source filename.
            tags: Optional tags.
            created_at: ISO-8601 creation timestamp.

        Returns:
            List of Chunk objects.
        """
        tags = tags or []
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        source_id = source_id or f"meeting_{compute_text_hash(str(transcript))[:12]}"

        # Accept both {"turns": [...]} and direct list
        turns = transcript if isinstance(transcript, list) else transcript.get("turns", [])

        chunks = []
        for idx, turn in enumerate(turns):
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "").strip()
            timestamp = turn.get("timestamp", "")

            if not text:
                continue

            # Format: "[timestamp] Speaker: text"
            if timestamp:
                chunk_text = f"[{timestamp}] {speaker}: {text}"
            else:
                chunk_text = f"{speaker}: {text}"

            text_hash = compute_text_hash(chunk_text)
            turn_tags = tags.copy()
            if speaker:
                turn_tags.append(f"speaker:{speaker}")

            metadata = ChunkMetadata(
                source_type="meeting",
                source_id=source_id,
                file_name=file_name,
                chunk_id=str(uuid.uuid4()),
                chunk_index=idx,
                chunk_type="meeting_transcript",
                section_title=f"Turn {idx + 1} — {speaker}",
                language="en",
                created_at=created_at,
                indexed_at="",
                version=1,
                hash=text_hash,
                tags=turn_tags,
            )
            chunks.append(Chunk(text=chunk_text, metadata=metadata))

        logger.info(f"Chunked transcript '{file_name}' → {len(chunks)} turns")
        return chunks

    def _split_into_sections(self, text: str) -> List[str]:
        """Split text at semantic section boundaries."""
        lines = text.split("\n")
        boundaries = _find_section_boundaries(text)

        if len(boundaries) <= 1:
            return [text]

        sections = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
            section = "\n".join(lines[start:end]).strip()
            if section:
                sections.append(section)

        return sections

    def _split_large_section(self, text: str) -> List[str]:
        """Split a section that exceeds max tokens into smaller pieces."""
        paragraphs = re.split(r"\n\s*\n", text)
        result = []
        current = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = estimate_tokens(para)

            # If a single paragraph exceeds max, split by sentences
            if para_tokens > self.max_tokens:
                if current:
                    result.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                result.extend(self._split_by_sentences(para))
                continue

            if current_tokens + para_tokens > self.target_tokens and current:
                result.append("\n\n".join(current))
                current = []
                current_tokens = 0

            current.append(para)
            current_tokens += para_tokens

        if current:
            result.append("\n\n".join(current))

        return result

    def _split_by_sentences(self, text: str) -> List[str]:
        """Last-resort split by sentences for very long paragraphs."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = estimate_tokens(sent)

            if current_tokens + sent_tokens > self.target_tokens and current:
                result.append(" ".join(current))
                current = []
                current_tokens = 0

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            result.append(" ".join(current))

        return result

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge consecutive chunks that are below minimum size."""
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for chunk in chunks[1:]:
            combined_tokens = estimate_tokens(current + "\n\n" + chunk)

            if estimate_tokens(current) < self.min_tokens and combined_tokens <= self.max_tokens:
                current = current + "\n\n" + chunk
            else:
                merged.append(current)
                current = chunk

        merged.append(current)
        return merged

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add token overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1]
            # Get last N tokens worth of text from previous chunk
            overlap_chars = self.overlap * 4  # ~4 chars per token
            overlap_text = prev_text[-overlap_chars:].strip()

            # Find a clean word boundary for the overlap
            space_idx = overlap_text.find(" ")
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]

            result.append(f"{overlap_text}\n\n{chunks[i]}")

        return result
