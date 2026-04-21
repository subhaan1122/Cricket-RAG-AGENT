# ICC Cricket World Cup Dataset (2003–2023)

## Overview
Complete ICC Cricket World Cup dataset spanning 2003–2023 (6 tournaments, 299 matches), purpose-built for Retrieval-Augmented Generation (RAG) applications. The dataset provides four complementary data layers — structured match data, deep statistical analysis, enriched metadata, and semantic embedding text — ensuring comprehensive coverage for match queries, player comparisons, team analysis, captain performance, and tournament-level insights.

**Total size:** ~7.7 MB across 912 files.

## Dataset Structure

```
Cricket Data/
├── cleaned_matches/          → 299 standardized match JSONs (2.15 MB)
├── statistical_analysis/     → 299 detailed analysis JSONs (4.26 MB)
├── metadata/                 → 8 metadata files (0.67 MB)
├── embeddings/               → 306 text files for RAG/vector search (0.65 MB)
└── README.md
```

---

## 1. Cleaned Matches (`cleaned_matches/` — 299 files)

Standardized JSON files with consistent schema across all tournaments.

### Schema

| Field | Description |
|-------|-------------|
| `match_id` | Filename-based ID (e.g., `2023_final_australia_vs_india`) |
| `world_cup_year` | 2003 / 2007 / 2011 / 2015 / 2019 / 2023 |
| `stage` | Group A–D, Pool A–B, Super Six, Super Eight, Quarter-Final, Semi-Final, Final |
| `venue`, `city`, `date` | Location and date of match |
| `teams` | `{team1, team2}` |
| `toss` | `{winner, decision}` |
| `result` | `{winner, margin, method, result_type}` |
| `captains` | `{team1: name, team2: name}` |
| `player_of_match` | POM name (292 filled; 7 empty = forfeits/no-results) |
| `data_source` | `ball_by_ball` (267) or `summary_only` (32) |
| `match_summary` | Natural language summary |

### Innings Array
Each innings contains:
- `team`, `runs`, `wickets`, `overs`, `extras`
- `top_batters[]` — name, runs, balls, fours, sixes, strike_rate, not_out
- `top_bowlers[]` — name, overs, maidens, runs_conceded, wickets, economy
- `fall_of_wickets[]` — batter, score, over
- `partnerships[]` — batter1, batter2, runs, balls, wicket

---

## 2. Statistical Analysis (`statistical_analysis/` — 299 files)

Deep statistical breakdowns computed from ball-by-ball data.

### Schema

| Section | Contents |
|---------|----------|
| `match_info` | Full match metadata |
| `match_summary` | Natural language summary |
| `statistical_analysis` | Total runs, wickets, run rate, highest score, best bowling, sixes, fours |
| `innings_analysis[]` | Per-innings: top batters/bowlers, fall of wickets, partnerships, **powerplay_stats**, **death_overs_stats** |
| `player_analysis` | Individual player batting/bowling in this match |
| `match_insights[]` | Auto-generated key insights (natural language) |

### Phase-of-Play Stats
For each innings (529 of 585 innings filled — 56 empty are summary-only matches):

```json
"powerplay_stats": {
    "overs": "10.0", "balls": 60, "runs": 80, "wickets": 2,
    "run_rate": 8.0, "fours": 9, "sixes": 3, "dot_balls": 28, "extras": 2
}
"death_overs_stats": {
    "overs": "10.0", "balls": 60, "runs": 43, "wickets": 5,
    "run_rate": 4.3, "fours": 2, "sixes": 0, "dot_balls": 30, "extras": 5
}
```

---

## 3. Metadata (`metadata/` — 8 files)

### `match_index.json` (299 entries)
Fast-lookup index with filename-based `match_id`, teams, stage, venue, winner, margin, POM, toss, and **captains** per match.

### `player_stats.json` (679 players)
Career aggregates across all World Cups:
- **Batting**: matches, runs, innings, not_outs, highest_score, average, strike_rate, centuries, fifties, fours, sixes
- **Bowling**: wickets, balls_bowled, runs_conceded, average, economy, best_bowling, maidens
- **Enriched fields**: `role` (Batter/Bowler/All-Rounder), `is_captain`, `captained` (list of {year, team})

### `{year}_world_cup_summary.json` (6 files)
Each tournament summary includes:
- Basic info: host, format, teams, winner, runner-up, semi-finalists
- Top performers: player_of_tournament, top_run_scorer, top_wicket_taker
- `team_standings[]` — played, won, lost, win percentage
- **`team_performance`** — per-team aggregates (runs scored/conceded, wickets, highest/lowest totals, win%)
- **`head_to_head`** — all team-pair matchups with wins and results
- **`captain_performance`** — per-captain stats (matches, wins, toss wins, batting runs, win%)

---

## 4. Embeddings (`embeddings/` — 306 files)

Richly structured text files optimized for vector search and RAG.

### Match Embeddings (299 files)
Each file contains these **section headings** for semantic chunking:

```
ICC Cricket World Cup {year}
Match: {team1} vs {team2}
Stage: / Date: / Venue: / Captain:

Match Summary:
  {brief result line}

{Innings 1}: {score}
  {batters with runs, balls, 4s, 6s, SR}
  {bowlers with wickets/runs, overs, economy}

{Innings 2}: {score}
  ...

Batting Highlights:
  Century: / Half-century: / Quickfire innings:

Bowling Highlights:
  Match-changing spell: / Key spell: / Economical spell:

Captain Performance:
  {captain batting/bowling contributions}

Key Moments:
  {powerplay collapses, death over surges, century partnerships}

Result: / Toss: / Player of the Match:
```

### Other Embeddings
- **6 tournament summary files** (`{year}_tournament_summary.txt`)
- **1 player statistics file** (`all_player_statistics.txt` — 679 players, 4224 lines)

---

## Data Sources

| Source | Matches | Coverage |
|--------|---------|----------|
| Cricsheet (ball-by-ball) | 267 | Full deliveries, partnerships, fall of wickets |
| Manual compilation (Wikipedia) | 32 | Score summaries (Afghanistan matches withheld by Cricsheet) |
| Forfeit/No-result | 7 | Minimal data (2 forfeits, 5 abandoned) |

---

## Data Quality

- **POM coverage**: 292/299 (7 empty = 2 forfeits + 5 no-results)
- **Powerplay stats**: 529/585 innings (56 empty = summary-only matches with no ball-by-ball data)
- **Captains**: 299/299 matches
- **Cross-validated**: All 4 folders are aligned (match IDs consistent across cleaned_matches, statistical_analysis, embeddings, and match_index)
- **Embedding sections**: All 299 match files have Match Summary, Batting Highlights, Bowling Highlights, Captain Performance, Key Moments

---

## Usage Examples

### Match Lookup
```python
import json
match = json.load(open("cleaned_matches/2023_final_australia_vs_india.json"))
print(f"Winner: {match['result']['winner']}")
print(f"POM: {match['player_of_match']}")
print(f"Captain: {match['captains']['team1']}")
```

### Powerplay Analysis
```python
stats = json.load(open("statistical_analysis/2023_final_australia_vs_india.json"))
pp = stats['innings_analysis'][0]['powerplay_stats']
print(f"Powerplay: {pp['runs']} runs, {pp['wickets']} wickets, RR {pp['run_rate']}")
```

### Player Comparison
```python
ps = json.load(open("metadata/player_stats.json"))
for name in ["V Kohli", "RT Ponting", "RG Sharma"]:
    p = ps[name]
    print(f"{name}: {p['runs']} runs, avg {p['batting_average']}, {p['centuries']} centuries, captain={p['is_captain']}")
```

### Captain Performance
```python
ts = json.load(open("metadata/2023_world_cup_summary.json"))
for cap, data in ts['captain_performance'].items():
    print(f"{cap} ({data['team']}): {data['wins']}/{data['matches_as_captain']} wins, {data['win_percentage']}%")
```

### Head-to-Head
```python
ts = json.load(open("metadata/2023_world_cup_summary.json"))
h2h = ts['head_to_head']['Australia vs India']
print(f"Matches: {h2h['matches']}, Wins: {h2h['wins']}")
```

### RAG / Vector Search
```python
with open("embeddings/2023_final_australia_vs_india.txt") as f:
    text = f.read()
# Split by section headings for semantic chunking
# Feed into vector database (Pinecone, Chroma, FAISS, etc.)
```

---

## Key Statistics

| Tournament | Matches | Winner | Top Scorer | Top Wicket-Taker |
|------------|---------|--------|------------|------------------|
| 2003 | 54 | Australia | SR Tendulkar (673) | B Lee (22) |
| 2007 | 51 | Australia | ML Hayden (659) | GD McGrath (26) |
| 2011 | 49 | India | TM Dilshan (500) | Zaheer Khan (21) |
| 2015 | 49 | Australia | KC Sangakkara (541) | MA Starc (22) |
| 2019 | 48 | England | RG Sharma (648) | MA Starc (27) |
| 2023 | 48 | Australia | V Kohli (765) | Mohammed Shami (24) |

**All-Time Leaders:**
- **Most Runs**: Virat Kohli (1,638)
- **Most Wickets**: Mitchell Starc (61)
- **Most Centuries**: Rohit Sharma (7)

---

## Supported Query Types

This dataset is optimized to answer:
- **Match queries**: "Who won the 2023 final?" / "What was the score?"
- **Player comparisons**: "Compare Kohli vs Ponting across World Cups"
- **Team analysis**: "India's win percentage in 2023" / "Head-to-head records"
- **Captain stats**: "Which captain has the best win rate?"
- **Phase analysis**: "Powerplay scoring rates" / "Death overs wickets"
- **Tournament overviews**: "2019 World Cup summary and standings"
- **Historical queries**: "All centuries scored in World Cup finals"

## Technical Notes

- **Python**: 3.8+ compatible
- **Encoding**: UTF-8
- **No external dependencies**: Standard library only (json, os, re)
- **Total size**: ~7.7 MB (912 files)

## Last Updated
2026-02-07 (Dataset complete, optimized, and verified)