# Cricket World Cup Dataset — Embedding Selection Guide for LLMs

## Overview
This guide helps you select the perfect embeddings from the Cricket World Cup dataset (2003-2023) for your Retrieval-Augmented Generation (RAG) applications. The dataset contains 306 embedding files across 4 folders, each optimized for different query types and use cases. Understanding what each file contains will help you build more accurate and relevant RAG systems.

## Dataset Structure Recap

```
Cricket Data/
├── cleaned_matches/          → 299 JSON files (structured match data)
├── statistical_analysis/     → 299 JSON files (detailed stats)
├── metadata/                 → 8 JSON files (indexes and aggregates)
├── embeddings/               → 306 TXT files (semantic text for RAG)
└── Embedding_Selection_Guide.md → This guide
```

## Embedding Files Overview

### Match Embeddings (299 files)
**Location:** `embeddings/{match_id}.txt`  
**Example:** `embeddings/2023_final_australia_vs_india.txt`  
**Purpose:** Individual match narratives for match-specific queries

#### What Each Match Embedding Contains:
Each file is structured with 5 semantic sections for optimal chunking:

1. **Match Header** (Lines 1-4)
   - Tournament year and match details
   - Teams, stage, date, venue, captains
   - Basic match information

2. **Match Summary** (Lines 5-6)
   - Brief result summary
   - Winner, margin, and key outcome

3. **Innings Details** (Lines 7-20)
   - Full scorecard for each innings
   - Top batters (runs, balls, boundaries, strike rate)
   - Top bowlers (wickets, runs, overs, economy)
   - Fall of wickets and partnerships

4. **Batting Highlights** (Lines 21-25)
   - Centuries and half-centuries
   - Quickfire innings (30+ runs in <20 balls)
   - Notable individual performances

5. **Bowling Highlights** (Lines 26-30)
   - Match-changing spells (3+ wickets)
   - Economical spells (<4 RPO)
   - Key bowling performances

6. **Captain Performance** (Lines 31-35)
   - Captain batting contributions
   - Captain bowling figures
   - Leadership impact

7. **Key Moments** (Lines 36-40)
   - Powerplay collapses or surges
   - Death over performances
   - Century partnerships
   - Turning points

8. **Result & Toss** (Lines 41-42)
   - Final result details
   - Toss winner and decision
   - Player of the Match

#### When to Use Match Embeddings:
- **Match-specific queries**: "What happened in the 2023 final?"
- **Player performance in a match**: "How did Virat Kohli perform in the 2019 semi-final?"
- **Scorecard lookups**: "What were the bowling figures in Australia vs India 2023?"
- **Match narratives**: "Tell me about the 2011 World Cup final"
- **Captain analysis per match**: "How did MS Dhoni captain in the 2011 final?"

### Tournament Summary Embeddings (6 files)
**Location:** `embeddings/{year}_tournament_summary.txt`  
**Example:** `embeddings/2023_tournament_summary.txt`  
**Purpose:** Complete tournament overviews and standings

#### What Each Tournament Embedding Contains:
- **Tournament Overview**: Host country, format, participating teams, winner
- **Final Standings**: Team rankings with played, won, lost, win percentage
- **Top Performers**: Player of tournament, top run scorer, top wicket taker
- **Team Performance**: Per-team aggregates (runs scored/conceded, wickets, totals)
- **Head-to-Head Records**: All team pair matchups with results
- **Captain Performance**: Per-captain stats (matches, wins, batting runs, win %)
- **Key Statistics**: Tournament totals, averages, records
- **Notable Achievements**: Centuries, five-wicket hauls, partnerships

#### When to Use Tournament Embeddings:
- **Tournament-level queries**: "Who won the 2019 World Cup?"
- **Team performance**: "How did India perform in the 2023 World Cup?"
- **Head-to-head analysis**: "Australia vs India record in World Cups"
- **Captain comparisons**: "Which captain had the best win rate in 2015?"
- **Tournament statistics**: "What were the batting averages in 2007?"

### Player Statistics Embedding (1 file)
**Location:** `embeddings/all_player_statistics.txt`  
**Purpose:** Comprehensive player career data across all tournaments

#### What the Player Statistics Embedding Contains:
- **679 Players**: All players who appeared in World Cups 2003-2023
- **Career Aggregates**: Total runs, wickets, averages, strike rates
- **Batting Stats**: Centuries, fifties, highest scores, boundaries
- **Bowling Stats**: Best figures, maidens, economy rates
- **Role Classification**: Batter, Bowler, All-Rounder
- **Captaincy Data**: Matches captained, wins as captain
- **Tournament Breakdown**: Performance by year
- **Comparative Rankings**: All-time leaders and records

#### When to Use Player Statistics Embedding:
- **Player comparisons**: "Compare Virat Kohli vs Sachin Tendulkar"
- **Career statistics**: "What is Mitchell Starc's bowling record?"
- **All-time leaders**: "Who has the most runs in World Cup history?"
- **Player roles**: "List all all-rounders in World Cups"
- **Captaincy analysis**: "Which captains have the most wins?"

## Choosing the Right Embeddings for Your RAG System

### Query Type → Recommended Embeddings

#### 1. Specific Match Questions
**Examples:** "Who won the 2019 final?" "What was the score in Australia vs India 2023?"
- **Use:** Individual match embeddings (`embeddings/{match_id}.txt`)
- **Why:** Contains complete match details, scorecards, and narratives
- **Chunking:** Split by section headings for precise retrieval

#### 2. Player Performance in Matches
**Examples:** "How did Rohit Sharma bat in the 2019 semi-final?" "What were Jasprit Bumrah's figures?"
- **Use:** Individual match embeddings + player statistics embedding
- **Why:** Match files have in-match performance; player stats have career context
- **Strategy:** Retrieve match embedding first, then enrich with career stats

#### 3. Tournament-Level Analysis
**Examples:** "What were the standings in 2015?" "Who was player of tournament in 2023?"
- **Use:** Tournament summary embeddings (`embeddings/{year}_tournament_summary.txt`)
- **Why:** Contains complete tournament data, standings, and aggregates
- **Chunking:** Large files - consider splitting by team or section

#### 4. Team Comparisons & Head-to-Head
**Examples:** "India vs Australia record" "Which team scored most runs in 2019?"
- **Use:** Tournament summary embeddings
- **Why:** Head-to-head records and team performance data
- **Strategy:** Query specific tournament files for historical context

#### 5. Captain Performance & Leadership
**Examples:** "Which captain has the best win rate?" "How did Virat Kohli captain in 2023?"
- **Use:** Tournament summary embeddings + match embeddings
- **Why:** Tournament files have captain aggregates; match files have per-match leadership
- **Strategy:** Use tournament files for overall stats, match files for specific instances

#### 6. Statistical Analysis & Records
**Examples:** "Most centuries in World Cups" "Highest individual score"
- **Use:** Player statistics embedding + tournament summary embeddings
- **Why:** Player stats has all-time records; tournament files have yearly records
- **Strategy:** Player stats for individual records, tournament files for yearly context

#### 7. Phase-of-Play Analysis
**Examples:** "Powerplay scoring rates" "Death overs performances"
- **Use:** Individual match embeddings (contain phase highlights)
- **Why:** Key moments section includes powerplay and death over analysis
- **Note:** For detailed stats, combine with statistical_analysis JSON files

#### 8. Historical Trends & Comparisons
**Examples:** "How has cricket changed from 2003 to 2023?" "Evolution of run rates"
- **Use:** All tournament summary embeddings + player statistics
- **Why:** Cross-tournament data for trend analysis
- **Strategy:** Multi-file retrieval across years

### Advanced RAG Strategies

#### Semantic Chunking Recommendations
- **Match Embeddings:** Split by the 5 section headings for precise retrieval
- **Tournament Embeddings:** Split by team sections or performance categories
- **Player Stats:** Split by player or statistical category

#### Hybrid Retrieval
- **Structured + Semantic:** Use JSON files (cleaned_matches, statistical_analysis) for exact facts, embeddings for narratives
- **Multi-Stage:** First retrieve from embeddings, then verify with structured data

#### Vector Database Optimization
- **Match Queries:** Index individual match embeddings separately
- **Tournament Queries:** Index tournament summaries as larger chunks
- **Player Queries:** Index player stats with metadata filters

## File Counts & Coverage

| Embedding Type | Count | Coverage | Best For |
|----------------|-------|----------|----------|
| Match Embeddings | 299 | All matches | Match-specific queries |
| Tournament Summaries | 6 | 2003-2023 | Tournament analysis |
| Player Statistics | 1 | All players | Career comparisons |
| **Total** | **306** | **Complete** | **All query types** |

## Quality Assurance

- **All embeddings validated:** 299 match files have all 5 sections
- **Data consistency:** Aligned with JSON files across all folders
- **POM coverage:** 292/299 matches have Player of Match data
- **Captain data:** All matches include captain information
- **Statistical accuracy:** Computed from ball-by-ball data where available

## Implementation Tips for LLMs

### Prompt Engineering
When asking LLMs about embeddings:
- "Which embedding files should I use for questions about specific matches?"
- "For tournament-level statistics, which files contain the relevant data?"
- "How are the embedding files structured for optimal retrieval?"

### Common Pitfalls to Avoid
- Don't use match embeddings for tournament questions (too granular)
- Don't use player stats for match-specific details (too aggregated)
- Always check data_source in JSON files (ball_by_ball vs summary_only)

### Performance Optimization
- **Pre-filtering:** Use metadata/match_index.json to identify relevant match_ids
- **Batch processing:** Load tournament embeddings for multi-match queries
- **Caching:** Cache frequently accessed player statistics

This guide ensures you select the most appropriate embeddings for your specific use case, maximizing the accuracy and relevance of your RAG cricket Q&A system.