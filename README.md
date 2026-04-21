# 🏏 Cricket World Cup RAG Assistant

A production-grade RAG (Retrieval-Augmented Generation) chatbot for ICC Cricket World Cup queries (2003–2023). Features **hybrid search** (FAISS semantic + BM25 keyword), **anti-hallucination** safeguards, and a **FastAPI + web frontend**.

## ✨ Features

- **🔍 Hybrid Search**: FAISS semantic search + BM25 keyword search with Reciprocal Rank Fusion (RRF)
- **🤖 AI-Powered Responses**: OpenRouter LLM with query-type-specific prompts and anti-hallucination rules
- **📚 Rich RAG Pipeline**: Multi-query retrieval, query rewriting, entity resolution, context re-ranking
- **🎯 Query Classification**: Automatic detection of statistical, comparative, match-specific, tournament, player, and general queries
- **💬 Two Interfaces**: Interactive CLI + FastAPI web server with polished HTML/CSS/JS frontend
- **🏏 Cricket-Aware**: Entity resolution for player nicknames, team abbreviations, tournament years
- **⚡ Local Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (free, no API costs)
- **📊 Rich Context Data**: Curated memorable moments, cross-tournament records, and detailed match data

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure OpenRouter
Create a `.env` file:
```env
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-3-haiku
LLM_API_KEY=sk-or-v1-xxxxxxxxxxxxx
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MAX_TOKENS=1500
LLM_TEMPERATURE=0.3
```

### 3. Build the Index (First Time Only)
```bash
python main.py --build-index
```

### 4. Start Using

**Option A — CLI Mode:**
```bash
python main.py
```

**Option B — Web Server + Frontend:**
```bash
python server.py
# Open http://localhost:8000 in your browser
```

**Option C — Single Question:**
```bash
python main.py --query "Who won the 2019 World Cup final?"
```

## 💻 CLI Commands

```
/status  — Show system status (vectors, chunks, LLM info)
/history — Show chat history from history.txt
/clear   — Clear conversation history
/build   — Rebuild FAISS + BM25 index
/help    — Show help
/quit    — Exit
```

## 🌐 API Endpoints (server.py)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a question, get a RAG-powered answer |
| `GET` | `/status` | System status and statistics |
| `POST` | `/build` | Rebuild the search index |
| `POST` | `/clear-history` | Clear conversation history |
| `GET` | `/health` | Simple health check |
| `GET` | `/` | Landing page (frontend) |
| `GET` | `/chat-page` | Chatbot page (frontend) |

### Example API Call
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Who scored the most runs in 2023 World Cup?"}'
```

## 🏗️ Architecture

```
Cricket-World-Cup-RAG-Assistant/
├── main.py                 # CLI chatbot, query pipeline, prompt templates
├── server.py               # FastAPI web server
├── embeddings_utils.py     # Orchestrator: search, context assembly, index management
├── config.py               # Central configuration (BM25, search, chunking params)
├── requirements.txt        # Python dependencies
├── Embedding/
│   ├── embeddings.py       # Sentence-transformer embedding generation
│   ├── vector_store.py     # FAISS + BM25 hybrid vector store
│   ├── chunking.py         # Cricket-aware text chunking engine
│   └── ingestion.py        # Data ingestion pipeline
├── Frontend/
│   ├── index.html          # Landing page
│   ├── chatbot.html        # Chat interface
│   ├── chatbot.js          # Chat logic (connects to /chat API)
│   ├── script.js           # Landing page logic
│   └── style.css           # Styles
├── Cricket Data/
│   ├── cleaned_matches/    # Match JSON files (2003–2023)
│   ├── embeddings/         # Pre-processed text + memorable moments + records
│   ├── metadata/           # Tournament metadata
│   └── statistical_analysis/ # Performance statistics
└── index/
    ├── faiss.index         # FAISS HNSW vector index
    ├── chunks.json         # Indexed text chunks
    ├── bm25_corpus.json    # BM25 keyword search corpus
    └── vectors.json        # Raw vectors
```

## 🔧 How It Works

### RAG Pipeline (per query)

1. **Query Rewriting** — Resolve pronouns, temporal references, expand abbreviations
2. **Query Classification** — Detect type (statistical / comparative / match / tournament / player / general)
3. **Entity Resolution** — Resolve player nicknames ("MSD" → "MS Dhoni"), team names ("Aussies" → "Australia")
4. **Sub-Query Generation** — Break complex queries into 5–20 targeted sub-queries
5. **Hybrid Search** — FAISS semantic search + BM25 keyword search, fused via RRF
6. **Context Assembly** — Merge, deduplicate, and truncate results within token limits
7. **Prompt Construction** — Query-type-specific system prompts with anti-hallucination rules
8. **LLM Generation** — OpenRouter API call with context + conversation history
9. **Response** — Structured answer with source attribution and metadata

### Hybrid Search Details

- **Semantic (FAISS HNSW)**: 384-dim embeddings from all-MiniLM-L6-v2
- **Keyword (BM25)**: rank-bm25 with cricket-aware tokenization and stopword removal
- **Fusion**: Reciprocal Rank Fusion with configurable weights per query type
  - Statistical queries: 60% semantic, 40% BM25
  - Match-specific: 65% semantic, 35% BM25
  - General: 75% semantic, 25% BM25

### Anti-Hallucination Safeguards

- Query-type-specific prompt instructions with accuracy warnings
- Explicit facts encoded in system prompt (e.g., 2019 final boundary countback details)
- Context-grounded answers — LLM instructed to trust context data over its own knowledge
- Coverage validation — detects when context may be insufficient

## ⚙️ Configuration

### Environment Variables (.env)

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider name | `openrouter` |
| `LLM_MODEL` | AI model to use | `anthropic/claude-3-haiku` |
| `LLM_API_KEY` | Your OpenRouter API key | `sk-or-v1-...` |
| `LLM_BASE_URL` | API base URL | `https://openrouter.ai/api/v1` |
| `LLM_MAX_TOKENS` | Max response tokens | `1500` |
| `LLM_TEMPERATURE` | Response creativity (0.0–1.0) | `0.3` |

### Key Config Parameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BM25_ENABLED` | `True` | Enable/disable BM25 hybrid search |
| `HYBRID_SEMANTIC_WEIGHT` | `0.7` | Default semantic search weight |
| `HYBRID_BM25_WEIGHT` | `0.3` | Default BM25 search weight |
| `MAX_CONTEXT_CHARS` | `14000` | Max characters in LLM context |
| `MAX_CONTEXT_CHUNKS` | `30` | Max chunks in context |
| `RERANKING_ENABLED` | `True` | Enable chunk re-ranking |
| `SEARCH_TOP_K_DEFAULT` | `10` | Default results per search |
| `SEARCH_SCORE_THRESHOLD` | `0.25` | Minimum similarity score |

### Supported OpenRouter Models

- `anthropic/claude-3-haiku` (fast, cost-effective)
- `anthropic/claude-3-sonnet` (balanced performance)
- `openai/gpt-4o-mini` (good balance)
- `meta-llama/llama-3.1-8b-instruct` (free tier)

## 🎯 Example Queries

- "Who won the 2019 World Cup final and how?"
- "Compare Virat Kohli and Ricky Ponting's World Cup records"
- "What happened in the 2003 World Cup final?"
- "Tell me about the biggest upsets in World Cup history"
- "Who scored the most centuries across all World Cups?"
- "Describe the 2007 World Cup Super Eight format"
- "What was MS Dhoni's World Cup career like?"
- "Which bowler had the best economy rate in 2015?"

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | `pip install -r requirements.txt` |
| API key not found | Check `.env` file exists with correct values |
| Knowledge base not initialized | Run `python main.py --build-index` |
| Poor answer quality | Try a stronger model, or rebuild index with `--build-index` |
| BM25 errors | Ensure `rank-bm25` is installed: `pip install rank-bm25` |
| Server won't start | Check port 8000 is free, try `python server.py` |

## 📈 System Requirements

- **Python**: 3.9+
- **RAM**: 4GB+ (for FAISS index + BM25 corpus)
- **Storage**: ~100MB (indexes, embeddings, data)
- **Internet**: Required for OpenRouter API calls only (embeddings are local)

## 📄 License

This project uses cricket data from public ICC sources. AI responses generated using your OpenRouter account.

---

**Ready to explore cricket history?** 🏏✨

```bash
python server.py
# Then open http://localhost:8000
```
#   C r i c k e t - R A G - A G E N T  
 