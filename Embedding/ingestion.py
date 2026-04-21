"""
Cricket World Cup RAG — Document Ingestion Pipeline
======================================================
Handles text extraction from multiple source types,
chunking, embedding generation, BM25 indexing, and FAISS indexing.

Supports: .txt, .json, .md, .csv, conversation.json
Deduplication: SHA-256 hash-based (same text → skip)
Incremental: Safe re-indexing without duplicated vectors
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .chunking import Chunk, ChunkMetadata, TextChunker, compute_text_hash
from config import (
    CHUNKS_JSON_PATH,
    DOCUMENTS_PROCESSED_DIR,
    DOCUMENTS_RAW_DIR,
    INDEX_DIR,
    METADATA_MD_PATH,
    INDEX_MANIFEST_PATH,
    CRICKET_EMBEDDINGS_DIR,
    CRICKET_METADATA_DIR,
    CRICKET_CLEANED_DIR,
    CRICKET_STATS_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class TextExtractor:
    """
    Extract text from various document formats.
    Keeps extraction logic centralized and extensible.
    """

    @staticmethod
    def extract_text(file_path: Path) -> str:
        """
        Extract plain text from a file based on its extension.

        Supported: .txt, .md, .json, .csv
        """
        suffix = file_path.suffix.lower()

        if suffix in (".txt", ".md"):
            return TextExtractor._read_text_file(file_path)
        elif suffix == ".json":
            return TextExtractor._extract_from_json(file_path)
        elif suffix == ".csv":
            return TextExtractor._extract_from_csv(file_path)
        else:
            # Attempt plain text read as fallback
            try:
                return TextExtractor._read_text_file(file_path)
            except Exception:
                logger.warning(f"Unsupported file type: {suffix} for {file_path}")
                return ""

    @staticmethod
    def _read_text_file(file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
    def _extract_from_json(file_path: Path) -> str:
        """
        Extract text from JSON files.
        Handles:
            - Cricket cleaned match JSONs → structured text
            - Cricket statistical analysis JSONs → structured text
            - World cup summary JSONs → rich tournament text
            - Player stats JSON → comprehensive player data
            - Match index JSON → match listing
            - Generic JSON → flattened key-value text
            - conversation.json → transcript text
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fname = file_path.name.lower()

        # Detect conversation/meeting transcript
        if isinstance(data, dict) and "turns" in data:
            return TextExtractor._transcript_to_text(data)
        if isinstance(data, list) and len(data) > 0 and "speaker" in data[0]:
            return TextExtractor._transcript_to_text({"turns": data})

        # Detect cricket cleaned match format
        if isinstance(data, dict) and "match_id" in data and "innings" in data:
            return TextExtractor._cricket_match_to_text(data)

        # Detect cricket statistical analysis format
        if isinstance(data, dict) and "match_info" in data and "statistical_analysis" in data:
            return TextExtractor._cricket_stats_to_text(data)

        # Detect world cup summary format
        if isinstance(data, dict) and "winner" in data and ("team_performance" in data or "team_standings" in data):
            return TextExtractor._extract_world_cup_summary(data)

        # Detect player_stats.json (large dict of player names → stats)
        if fname == "player_stats.json" and isinstance(data, dict):
            first_val = next(iter(data.values()), None) if data else None
            if isinstance(first_val, dict) and "runs" in first_val:
                return TextExtractor._extract_player_stats(data)

        # Detect match_index.json
        if fname == "match_index.json":
            return TextExtractor._extract_match_index(data)

        # Generic JSON → structured text
        return TextExtractor._flatten_json(data)

    @staticmethod
    def _extract_from_csv(file_path: Path) -> str:
        """Read CSV and convert to text table."""
        lines = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                lines.append(line.strip())
        return "\n".join(lines)

    @staticmethod
    def _transcript_to_text(data: dict) -> str:
        """Convert conversation.json to text."""
        turns = data.get("turns", [])
        lines = []
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            timestamp = turn.get("timestamp", "")
            if timestamp:
                lines.append(f"[{timestamp}] {speaker}: {text}")
            else:
                lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _cricket_match_to_text(data: dict) -> str:
        """Convert cleaned cricket match JSON to text."""
        parts = []
        parts.append(f"ICC Cricket World Cup {data.get('world_cup_year', '')}")
        teams = data.get("teams", {})
        t1 = teams.get("team1", "")
        t2 = teams.get("team2", "")
        parts.append(f"Match: {t1} vs {t2}")
        parts.append(f"Stage: {data.get('stage', '')}")
        parts.append(f"Date: {data.get('date', '')}")
        parts.append(f"Venue: {data.get('venue', '')}, {data.get('city', '')}")

        result = data.get("result", {})
        parts.append(f"Result: {result.get('winner', 'N/A')} won by {result.get('margin', 'N/A')}")
        parts.append(f"Player of the Match: {data.get('player_of_match', 'N/A')}")

        summary = data.get("match_summary", "")
        if summary:
            parts.append(f"\nMatch Summary:\n{summary}")

        # Innings
        for inn in data.get("innings", []):
            team = inn.get("team", "")
            runs = inn.get("runs", 0)
            wickets = inn.get("wickets", 0)
            overs = inn.get("overs", "")
            parts.append(f"\n{team}: {runs}/{wickets} ({overs} ov)")

            for bat in inn.get("top_batters", []):
                parts.append(
                    f"  {bat.get('name', '')}: {bat.get('runs', 0)} "
                    f"({bat.get('balls', 0)}b, {bat.get('fours', 0)}x4, "
                    f"{bat.get('sixes', 0)}x6, SR {bat.get('strike_rate', 0)})"
                )
            for bowl in inn.get("top_bowlers", []):
                parts.append(
                    f"  {bowl.get('name', '')}: {bowl.get('wickets', 0)}/"
                    f"{bowl.get('runs_conceded', 0)} ({bowl.get('overs', '')} ov, "
                    f"Econ {bowl.get('economy', 0)})"
                )

        return "\n".join(parts)

    @staticmethod
    def _cricket_stats_to_text(data: dict) -> str:
        """Convert statistical analysis JSON to comprehensive text."""
        parts = []
        mi = data.get("match_info", {})
        match_id = mi.get("match_id", "")
        year = mi.get("world_cup_year", "")
        teams = mi.get("teams", {})
        t1 = teams.get("team1", "") if isinstance(teams, dict) else str(teams)
        t2 = teams.get("team2", "") if isinstance(teams, dict) else ""
        parts.append(f"Statistical Analysis: {match_id}")
        parts.append(f"ICC Cricket World Cup {year}")
        parts.append(f"Match: {t1} vs {t2}")
        parts.append(f"Stage: {mi.get('stage', '')}")
        parts.append(f"Venue: {mi.get('venue', '')}, {mi.get('city', '')}")
        parts.append(f"Date: {mi.get('date', '')}")

        result = mi.get("result", {})
        if isinstance(result, dict):
            parts.append(f"Result: {result.get('winner', '')} won by {result.get('margin', '')}")
        parts.append(f"Player of the Match: {mi.get('player_of_match', '')}")

        summary = data.get("match_summary", "")
        if summary:
            parts.append(f"\nMatch Summary:\n{summary}")

        sa = data.get("statistical_analysis", {})
        if sa:
            parts.append(f"\nOverall Statistics:")
            parts.append(f"  Total runs: {sa.get('total_runs', 0)}")
            parts.append(f"  Total wickets: {sa.get('total_wickets', 0)}")
            parts.append(f"  Overall run rate: {sa.get('overall_run_rate', '')}")
            parts.append(f"  Highest score: {sa.get('highest_individual_score', '')}")
            parts.append(f"  Best bowling: {sa.get('best_bowling_figures', '')}")
            parts.append(f"  Total sixes: {sa.get('total_sixes', 0)}")
            parts.append(f"  Total fours: {sa.get('total_fours', 0)}")

        # Innings Analysis
        for inn in data.get("innings_analysis", []):
            team = inn.get("team", "")
            parts.append(f"\nInnings Analysis — {team}:")

            for bat in inn.get("top_batters", []):
                parts.append(
                    f"  Batting: {bat.get('name', '')} — {bat.get('runs', 0)} runs "
                    f"({bat.get('balls', 0)}b), SR {bat.get('strike_rate', 0)}, "
                    f"{bat.get('fours', 0)}x4, {bat.get('sixes', 0)}x6"
                )
            for bowl in inn.get("top_bowlers", []):
                parts.append(
                    f"  Bowling: {bowl.get('name', '')} — {bowl.get('wickets', 0)}/{bowl.get('runs_conceded', 0)} "
                    f"({bowl.get('overs', '')} ov), econ {bowl.get('economy', 0)}"
                )

            pp = inn.get("powerplay_stats", {})
            if pp and pp.get("runs"):
                parts.append(
                    f"  Powerplay: {pp.get('runs', 0)} runs, {pp.get('wickets', 0)} wickets, "
                    f"RR {pp.get('run_rate', 0)}, {pp.get('fours', 0)}x4, {pp.get('sixes', 0)}x6"
                )
            do = inn.get("death_overs_stats", {})
            if do and do.get("runs"):
                parts.append(
                    f"  Death Overs: {do.get('runs', 0)} runs, {do.get('wickets', 0)} wickets, "
                    f"RR {do.get('run_rate', 0)}"
                )

        # Player Analysis
        pa = data.get("player_analysis", {})
        if pa:
            parts.append("\nPlayer Analysis:")
            for player, pstats in pa.items():
                bat = pstats.get("batting", {})
                bowl = pstats.get("bowling", {})
                if bat and bat.get("runs", 0) > 0:
                    parts.append(
                        f"  {player} batting: {bat.get('runs', 0)} runs "
                        f"({bat.get('balls', 0)}b), SR {bat.get('strike_rate', 0)}"
                    )
                if bowl and bowl.get("wickets", 0) > 0:
                    parts.append(
                        f"  {player} bowling: {bowl.get('wickets', 0)}/{bowl.get('runs_conceded', 0)} "
                        f"({bowl.get('overs', '')} ov)"
                    )

        insights = data.get("match_insights", [])
        if insights:
            parts.append("\nKey Insights:")
            for insight in insights:
                parts.append(f"  - {insight}")

        return "\n".join(parts)

    @staticmethod
    def _extract_world_cup_summary(data: dict) -> str:
        """Convert world cup summary JSON to rich, searchable text."""
        parts = []
        year = data.get("year", "")
        parts.append(f"ICC Cricket World Cup {year} — Tournament Summary")
        parts.append(f"Host: {', '.join(data.get('host_countries', []))}")
        parts.append(f"Format: {data.get('format', '')}")
        parts.append(f"Total Matches: {data.get('total_matches', '')}")
        parts.append(f"Winner: {data.get('winner', '')}")
        parts.append(f"Runner-Up: {data.get('runner_up', '')}")
        parts.append(f"Semi-Finalists: {', '.join(data.get('semi_finalists', []))}")
        parts.append(f"Player of Tournament: {data.get('player_of_tournament', '')}")

        trs = data.get("top_run_scorer", {})
        if trs:
            parts.append(f"Top Run Scorer: {trs.get('name', '')} ({trs.get('runs', '')} runs)")
        twt = data.get("top_wicket_taker", {})
        if twt:
            parts.append(f"Top Wicket Taker: {twt.get('name', '')} ({twt.get('wickets', '')} wickets)")
        his = data.get("highest_individual_score", {})
        if his:
            parts.append(f"Highest Individual Score: {his.get('name', '')} — {his.get('runs', '')} runs vs {his.get('against', '')} ({his.get('match_stage', '')})")

        # Captains
        captains = data.get("captains", {})
        if captains:
            parts.append(f"\nCaptains in {year} World Cup:")
            for team, captain in captains.items():
                parts.append(f"  {team}: {captain}")

        # Team Standings
        standings = data.get("team_standings", [])
        if standings:
            parts.append(f"\nTeam Standings — {year} World Cup:")
            parts.append("  Team | Played | Won | Lost | Win%")
            for team in standings:
                name = team.get("team", "")
                played = team.get("played", team.get("matches", ""))
                won = team.get("won", team.get("wins", ""))
                lost = team.get("lost", team.get("losses", ""))
                win_pct = team.get("win_percentage", team.get("win_pct", ""))
                parts.append(f"  {name} | {played} | {won} | {lost} | {win_pct}%")

        # Team Performance
        tp = data.get("team_performance", {})
        if tp:
            parts.append(f"\nTeam Performance — {year} World Cup:")
            for team, stats in tp.items():
                matches = stats.get("matches", "")
                wins = stats.get("wins", "")
                losses = stats.get("losses", "")
                runs_scored = stats.get("runs_scored", "")
                runs_conceded = stats.get("runs_conceded", "")
                wickets_taken = stats.get("wickets_taken", "")
                highest = stats.get("highest_total", "")
                lowest = stats.get("lowest_total", "")
                win_pct = stats.get("win_percentage", "")
                parts.append(
                    f"  {team}: {matches} matches, {wins} wins, {losses} losses, "
                    f"win% {win_pct}%, runs scored {runs_scored}, runs conceded {runs_conceded}, "
                    f"wickets taken {wickets_taken}, highest total {highest}, lowest total {lowest}"
                )

        # Head to Head
        h2h = data.get("head_to_head", {})
        if h2h:
            parts.append(f"\nHead-to-Head Records — {year} World Cup:")
            for matchup, record in h2h.items():
                matches = record.get("matches", "")
                wins = record.get("wins", {})
                wins_str = ", ".join(f"{t}: {w}" for t, w in wins.items()) if isinstance(wins, dict) else str(wins)
                results = record.get("results", [])
                results_str = "; ".join(results) if results else ""
                parts.append(f"  {matchup}: {matches} match(es), Wins: {wins_str}. Results: {results_str}")

        # Captain Performance
        cp = data.get("captain_performance", {})
        if cp:
            parts.append(f"\nCaptain Performance — {year} World Cup:")
            for captain, stats in cp.items():
                team = stats.get("team", "")
                matches = stats.get("matches_as_captain", "")
                wins = stats.get("wins", "")
                losses = stats.get("losses", "")
                toss_wins = stats.get("toss_wins", "")
                batting_runs = stats.get("batting_runs", "")
                win_pct = stats.get("win_percentage", "")
                parts.append(
                    f"  {captain} ({team}): {matches} matches as captain, {wins} wins, "
                    f"{losses} losses, win% {win_pct}%, toss wins {toss_wins}, "
                    f"batting runs {batting_runs}"
                )

        return "\n".join(parts)

    @staticmethod
    def _extract_player_stats(data: dict) -> str:
        """Convert player_stats.json to rich, searchable text organized by player."""
        parts = []
        parts.append("ICC Cricket World Cup — All Player Statistics (2003-2023)")
        parts.append(f"Total Players: {len(data)}")
        parts.append("")

        for player_name, stats in data.items():
            teams = ", ".join(stats.get("teams", []))
            world_cups = ", ".join(str(y) for y in stats.get("world_cups", []))
            role = stats.get("role", "Unknown")
            matches = stats.get("matches", 0)
            runs = stats.get("runs", 0)
            innings = stats.get("innings_batted", 0)
            not_outs = stats.get("not_outs", 0)
            highest = stats.get("highest_score", 0)
            avg = stats.get("batting_average", 0)
            sr = stats.get("batting_strike_rate", 0)
            centuries = stats.get("centuries", 0)
            fifties = stats.get("fifties", 0)
            fours = stats.get("fours", 0)
            sixes = stats.get("sixes", 0)

            wickets = stats.get("wickets", 0)
            balls_bowled = stats.get("balls_bowled", 0)
            runs_conceded = stats.get("runs_conceded", 0)
            bowl_avg = stats.get("bowling_average", None)
            bowl_econ = stats.get("bowling_economy", None)
            best_bowling = stats.get("best_bowling", None)
            maidens = stats.get("maidens", 0)

            is_captain = stats.get("is_captain", False)
            captained = stats.get("captained", [])

            line = (
                f"Player: {player_name} | Team: {teams} | Role: {role} | "
                f"World Cups: {world_cups} | Matches: {matches} | "
                f"Batting: {runs} runs, {innings} innings, {not_outs} not outs, "
                f"HS {highest}, avg {avg}, SR {sr}, "
                f"{centuries} centuries, {fifties} fifties, {fours} fours, {sixes} sixes"
            )

            if wickets > 0 or balls_bowled > 0:
                bowl_str = (
                    f" | Bowling: {wickets} wickets, avg {bowl_avg}, "
                    f"econ {bowl_econ}, best {best_bowling}, {maidens} maidens"
                )
                line += bowl_str

            if is_captain and captained:
                cap_years = ", ".join(f"{c.get('year', '')} ({c.get('team', '')})" for c in captained)
                line += f" | Captain: {cap_years}"

            parts.append(line)

        return "\n".join(parts)

    @staticmethod
    def _extract_match_index(data: dict) -> str:
        """Convert match_index.json to searchable text."""
        parts = []
        parts.append("ICC Cricket World Cup — Match Index (2003-2023)")

        if isinstance(data, list):
            matches = data
        elif isinstance(data, dict):
            matches = list(data.values()) if not isinstance(list(data.values())[0] if data else None, (str, int, float)) else [data]
        else:
            return str(data)

        parts.append(f"Total Matches: {len(matches)}")
        parts.append("")

        for match in matches:
            if not isinstance(match, dict):
                continue
            match_id = match.get("match_id", "")
            teams = match.get("teams", {})
            t1 = teams.get("team1", "") if isinstance(teams, dict) else ""
            t2 = teams.get("team2", "") if isinstance(teams, dict) else ""
            stage = match.get("stage", "")
            year = match.get("world_cup_year", "")
            winner = match.get("winner", match.get("result", {}).get("winner", ""))
            margin = match.get("margin", match.get("result", {}).get("margin", ""))
            pom = match.get("player_of_match", match.get("pom", ""))
            venue = match.get("venue", "")
            captains = match.get("captains", {})
            cap1 = captains.get("team1", "") if isinstance(captains, dict) else ""
            cap2 = captains.get("team2", "") if isinstance(captains, dict) else ""

            parts.append(
                f"Match: {t1} vs {t2} | Year: {year} | Stage: {stage} | "
                f"Winner: {winner} by {margin} | POM: {pom} | Venue: {venue} | "
                f"Captains: {cap1}, {cap2}"
            )

        return "\n".join(parts)

    @staticmethod
    def _flatten_json(data, prefix: str = "") -> str:
        """Recursively flatten JSON to key: value text."""
        lines = []
        if isinstance(data, dict):
            for k, v in data.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    lines.append(TextExtractor._flatten_json(v, key))
                else:
                    lines.append(f"{key}: {v}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                key = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    lines.append(TextExtractor._flatten_json(item, key))
                else:
                    lines.append(f"{key}: {item}")
        else:
            lines.append(f"{prefix}: {data}")
        return "\n".join(filter(None, lines))


class IngestionPipeline:
    """
    Document ingestion pipeline.

    Flow:
        1. Load documents from raw/ directory (or Cricket Data)
        2. Extract text
        3. Chunk content
        4. Generate SHA-256 hash for each chunk
        5. Skip duplicates (hash-based)
        6. Generate embeddings via OpenAI
        7. Insert vectors into FAISS
        8. Persist: FAISS index, chunk map, vector map, metadata.md
        9. Update index_manifest.md
    """

    def __init__(self, embedding_generator, vector_store, chunker=None):
        """
        Args:
            embedding_generator: EmbeddingGenerator instance.
            vector_store: FAISSVectorStore instance.
            chunker: TextChunker instance (default created if None).
        """
        self._embedder = embedding_generator
        self._store = vector_store
        self._chunker = chunker or TextChunker()
        self._extractor = TextExtractor()

        # Chunk storage: chunk_id → Chunk
        self._chunks: Dict[str, Chunk] = {}
        # Hash set for deduplication
        self._indexed_hashes: Set[str] = set()

        # Load existing chunks if available
        self._load_existing_chunks()

    def _load_existing_chunks(self) -> None:
        """Load existing chunks from chunks.json for deduplication."""
        if CHUNKS_JSON_PATH.exists():
            try:
                with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for chunk_data in data.get("chunks", []):
                    chunk = Chunk.from_dict(chunk_data)
                    self._chunks[chunk.chunk_id] = chunk
                    self._indexed_hashes.add(chunk.hash)
                logger.info(f"Loaded {len(self._chunks)} existing chunks")
            except Exception as e:
                logger.warning(f"Could not load existing chunks: {e}")

    def ingest_directory(
        self,
        directory: Path,
        source_type: str = "document",
        tags: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> Dict[str, int]:
        """
        Ingest all supported files from a directory.

        Args:
            directory: Path to directory containing documents.
            source_type: Source type for all files.
            tags: Tags to apply to all chunks.
            recursive: Whether to recurse into subdirectories.

        Returns:
            Dict with ingestion statistics.
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return {"error": f"Directory not found: {directory}"}

        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_created": 0,
            "chunks_skipped_duplicate": 0,
            "vectors_added": 0,
            "errors": 0,
        }

        # Collect files
        if recursive:
            files = sorted(directory.rglob("*"))
        else:
            files = sorted(directory.iterdir())

        files = [f for f in files if f.is_file() and f.suffix.lower() in (
            ".txt", ".md", ".json", ".csv"
        )]

        logger.info(f"Ingesting {len(files)} files from {directory}")

        for file_path in files:
            try:
                result = self.ingest_file(
                    file_path=file_path,
                    source_type=source_type,
                    tags=tags,
                )
                stats["files_processed"] += 1
                stats["chunks_created"] += result.get("chunks_created", 0)
                stats["chunks_skipped_duplicate"] += result.get("chunks_skipped", 0)
                stats["vectors_added"] += result.get("vectors_added", 0)
            except Exception as e:
                stats["errors"] += 1
                stats["files_skipped"] += 1
                logger.error(f"Failed to ingest {file_path}: {e}")

        # Persist everything
        self._persist_all()

        logger.info(
            f"Ingestion complete: {stats['files_processed']} files, "
            f"{stats['chunks_created']} new chunks, "
            f"{stats['chunks_skipped_duplicate']} duplicates skipped, "
            f"{stats['vectors_added']} vectors added"
        )
        return stats

    def ingest_file(
        self,
        file_path: Path,
        source_type: str = "document",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Ingest a single file.

        Returns:
            Dict with per-file ingestion stats.
        """
        file_path = Path(file_path)
        tags = tags or []

        # Extract text
        text = self._extractor.extract_text(file_path)
        if not text.strip():
            logger.warning(f"Empty text extracted from {file_path}")
            return {"chunks_created": 0, "chunks_skipped": 0, "vectors_added": 0}

        # Detect if it's a meeting transcript
        if file_path.name == "conversation.json":
            return self._ingest_transcript(file_path, tags)

        # Chunk
        source_id = compute_text_hash(str(file_path))[:16]
        chunks = self._chunker.chunk_text(
            text=text,
            source_type=source_type,
            source_id=source_id,
            file_name=file_path.name,
            tags=tags,
        )

        return self._index_chunks(chunks)

    def _ingest_transcript(
        self, file_path: Path, tags: List[str]
    ) -> Dict[str, int]:
        """Ingest a conversation.json meeting transcript."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_id = compute_text_hash(str(file_path))[:16]
        chunks = self._chunker.chunk_meeting_transcript(
            transcript=data,
            source_id=source_id,
            file_name=file_path.name,
            tags=tags,
        )
        return self._index_chunks(chunks)

    def _index_chunks(self, chunks: List[Chunk]) -> Dict[str, int]:
        """
        Deduplicate, embed, and index chunks.

        Args:
            chunks: List of Chunk objects to index.

        Returns:
            Dict with stats.
        """
        now = datetime.now(timezone.utc).isoformat()
        new_chunks = []
        skipped = 0

        # Deduplicate by hash
        for chunk in chunks:
            if chunk.hash in self._indexed_hashes:
                skipped += 1
                continue
            chunk.metadata.indexed_at = now
            new_chunks.append(chunk)
            self._indexed_hashes.add(chunk.hash)

        if not new_chunks:
            return {"chunks_created": 0, "chunks_skipped": skipped, "vectors_added": 0}

        # Generate embeddings
        texts = [c.text for c in new_chunks]
        vectors = self._embedder.embed_batch(texts)

        # Add to FAISS + BM25
        chunk_ids = [c.chunk_id for c in new_chunks]
        self._store.add_vectors(vectors, chunk_ids, chunk_texts=texts)

        # Store chunks
        for chunk in new_chunks:
            self._chunks[chunk.chunk_id] = chunk

        return {
            "chunks_created": len(new_chunks),
            "chunks_skipped": skipped,
            "vectors_added": len(new_chunks),
        }

    def ingest_cricket_dataset(self, tags: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Ingest the entire Cricket World Cup dataset.
        Uses the pre-built embedding text files for optimal chunking.

        Returns:
            Ingestion statistics.
        """
        tags = tags or ["cricket", "world_cup"]
        total_stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "chunks_skipped_duplicate": 0,
            "vectors_added": 0,
            "errors": 0,
        }

        # 1. Ingest embedding text files (299 match + 6 tournament + 1 player)
        if CRICKET_EMBEDDINGS_DIR.exists():
            logger.info("Ingesting Cricket embedding text files...")
            for txt_file in sorted(CRICKET_EMBEDDINGS_DIR.glob("*.txt")):
                try:
                    # Determine source type based on filename
                    fname = txt_file.stem
                    file_tags = tags.copy()

                    if "tournament_summary" in fname:
                        src_type = "document"
                        file_tags.append("tournament_summary")
                    elif fname == "all_player_statistics":
                        src_type = "document"
                        file_tags.append("player_statistics")
                    elif "memorable_moments" in fname:
                        src_type = "document"
                        file_tags.append("memorable_moments")
                        # Extract year from filename
                        year = fname[:4]
                        if year.isdigit():
                            file_tags.append(f"year:{year}")
                    elif "cross_tournament" in fname or "records_and_facts" in fname:
                        src_type = "document"
                        file_tags.append("cross_tournament")
                        file_tags.append("records_and_facts")
                    elif "questions_answers" in fname or "cricket_questions" in fname:
                        src_type = "document"
                        file_tags.append("qa_pair")
                    elif "world_cup_summary" in fname:
                        src_type = "document"
                        file_tags.append("tournament_summary")
                    else:
                        src_type = "document"
                        file_tags.append("match")
                        # Extract year
                        year = fname[:4]
                        if year.isdigit():
                            file_tags.append(f"year:{year}")

                    result = self.ingest_file(
                        file_path=txt_file,
                        source_type=src_type,
                        tags=file_tags,
                    )
                    total_stats["files_processed"] += 1
                    total_stats["chunks_created"] += result.get("chunks_created", 0)
                    total_stats["chunks_skipped_duplicate"] += result.get("chunks_skipped", 0)
                    total_stats["vectors_added"] += result.get("vectors_added", 0)
                except Exception as e:
                    total_stats["errors"] += 1
                    logger.error(f"Failed to ingest {txt_file.name}: {e}")

        # 2. Ingest metadata files
        if CRICKET_METADATA_DIR.exists():
            logger.info("Ingesting Cricket metadata files...")
            for meta_file in sorted(CRICKET_METADATA_DIR.glob("*.json")):
                try:
                    result = self.ingest_file(
                        file_path=meta_file,
                        source_type="document",
                        tags=tags + ["metadata"],
                    )
                    total_stats["files_processed"] += 1
                    total_stats["chunks_created"] += result.get("chunks_created", 0)
                    total_stats["chunks_skipped_duplicate"] += result.get("chunks_skipped", 0)
                    total_stats["vectors_added"] += result.get("vectors_added", 0)
                except Exception as e:
                    total_stats["errors"] += 1
                    logger.error(f"Failed to ingest {meta_file.name}: {e}")

        # Persist everything
        self._persist_all()

        logger.info(
            f"Cricket dataset ingestion complete: "
            f"{total_stats['files_processed']} files, "
            f"{total_stats['chunks_created']} chunks, "
            f"{total_stats['vectors_added']} vectors"
        )
        return total_stats

    def _persist_all(self) -> None:
        """Persist index, chunks, and generate documentation."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        self._store.save()

        # Save chunks.json
        self._save_chunks()

        # Generate metadata.md
        self._generate_metadata_md()

        # Generate index_manifest.md
        self._generate_index_manifest()

    def _save_chunks(self) -> None:
        """Save all chunks to chunks.json with atomic write."""
        tmp_path = CHUNKS_JSON_PATH.with_suffix(".json.tmp")
        try:
            data = {
                "total_chunks": len(self._chunks),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "chunks": [c.to_dict() for c in self._chunks.values()],
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            shutil.move(str(tmp_path), str(CHUNKS_JSON_PATH))
            logger.info(f"Saved {len(self._chunks)} chunks to {CHUNKS_JSON_PATH}")
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            logger.error(f"Failed to save chunks: {e}")
            raise

    def _generate_metadata_md(self) -> None:
        """Generate human-readable metadata.md."""
        now = datetime.now(timezone.utc).isoformat()

        # Aggregate metadata
        source_types: Dict[str, int] = {}
        chunk_types: Dict[str, int] = {}
        files: Dict[str, int] = {}
        tag_counts: Dict[str, int] = {}

        for chunk in self._chunks.values():
            st = chunk.metadata.source_type
            source_types[st] = source_types.get(st, 0) + 1

            ct = chunk.metadata.chunk_type
            chunk_types[ct] = chunk_types.get(ct, 0) + 1

            fn = chunk.metadata.file_name
            files[fn] = files.get(fn, 0) + 1

            for tag in chunk.metadata.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        lines = [
            "# DIE Knowledge Base — Metadata",
            "",
            f"**Generated:** {now}",
            f"**Total Chunks:** {len(self._chunks)}",
            f"**Total Vectors:** {self._store.total_vectors}",
            "",
            "## Source Types",
            "",
            "| Source Type | Chunks |",
            "|------------|--------|",
        ]
        for st, count in sorted(source_types.items()):
            lines.append(f"| {st} | {count} |")

        lines.extend([
            "",
            "## Chunk Types",
            "",
            "| Chunk Type | Count |",
            "|------------|-------|",
        ])
        for ct, count in sorted(chunk_types.items()):
            lines.append(f"| {ct} | {count} |")

        lines.extend([
            "",
            "## Files Indexed",
            "",
            "| File | Chunks |",
            "|------|--------|",
        ])
        for fn, count in sorted(files.items()):
            lines.append(f"| {fn} | {count} |")

        lines.extend([
            "",
            "## Tags",
            "",
            "| Tag | Count |",
            "|-----|-------|",
        ])
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:30]:
            lines.append(f"| {tag} | {count} |")

        lines.append("")

        content = "\n".join(lines)
        with open(METADATA_MD_PATH, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Generated {METADATA_MD_PATH}")

    def _generate_index_manifest(self) -> None:
        """Generate index_manifest.md with index configuration and stats."""
        now = datetime.now(timezone.utc).isoformat()

        # Count unique documents
        unique_files = set(c.metadata.file_name for c in self._chunks.values())

        content = f"""# DIE Knowledge Base Index

- Embedding Model: {EMBEDDING_MODEL}
- Vector Dimension: {EMBEDDING_DIMENSION}
- Index Type: HNSWFlat
- Distance Metric: Cosine
- Total Documents: {len(unique_files)}
- Total Chunks: {len(self._chunks)}
- Total Vectors: {self._store.total_vectors}
- Last Indexed: {now}
- Version: v1.0

## Index Configuration

| Parameter | Value |
|-----------|-------|
| HNSW M | {self._store._hnsw_m} |
| EF Construction | {self._store._ef_construction} |
| EF Search | {self._store._ef_search} |
| Chunk Size (target) | 200-400 tokens |
| Chunk Overlap | 30 tokens |
| Deduplication | SHA-256 hash-based |

## Deduplication

- Unique content hashes: {len(self._indexed_hashes)}
- Guarantees: Safe re-indexing, no duplicated vectors, incremental updates

## Storage

| File | Purpose |
|------|---------|
| faiss.index | Binary FAISS vector index |
| chunks.json | Chunk ID → text + metadata mapping |
| vectors.json | Vector ID → chunk ID mapping |
| metadata.md | Human-readable metadata summary |
| index_manifest.md | This file — index config & stats |
"""

        with open(INDEX_MANIFEST_PATH, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Generated {INDEX_MANIFEST_PATH}")

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a chunk by ID."""
        return self._chunks.get(chunk_id)

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """Retrieve multiple chunks by IDs."""
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def get_all_chunks(self) -> Dict[str, Chunk]:
        """Get all indexed chunks."""
        return dict(self._chunks)

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def total_unique_hashes(self) -> int:
        return len(self._indexed_hashes)
