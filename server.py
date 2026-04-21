"""
FastAPI Server for Cricket World Cup RAG Chatbot
=================================================
Production-ready API server that serves the RAG chatbot and
static frontend files.

Endpoints:
    POST /chat          — Send a question, get an answer
    GET  /status        — Get system status and stats
    POST /build         — Rebuild the FAISS + BM25 index
    POST /clear-history — Clear conversation history
    GET  /health        — Simple health check

Run:
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from main import CricketChatbot

# ────────────────────────────────────────────────────────────
# LOGGING
# ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server")

# Suppress noisy HTTP logs from model loading
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# ────────────────────────────────────────────────────────────
# CHATBOT SINGLETON
# ────────────────────────────────────────────────────────────

chatbot = CricketChatbot()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize chatbot on startup, cleanup on shutdown."""
    logger.info("🏏 Starting Cricket World Cup RAG Server...")
    try:
        chatbot.initialize()
        logger.info("✅ Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {e}")
        raise
    yield
    logger.info("🏏 Shutting down Cricket World Cup RAG Server...")


# ────────────────────────────────────────────────────────────
# FASTAPI APP
# ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cricket World Cup RAG Chatbot",
    description="AI-powered chatbot for ICC Cricket World Cup queries (2003–2023)",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — explicit allowed origins (do not add '*' here when allow_credentials=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Add your production domain here, e.g.: "https://your-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The cricket question to ask",
        examples=["Who won the 2011 World Cup?"],
    )


class ChatResponse(BaseModel):
    """Response body for /chat endpoint."""
    answer: str
    query_type: str
    sources: list
    search_results: int
    processing_time: float


class BuildRequest(BaseModel):
    """Request body for /build endpoint."""
    force_rebuild: bool = Field(
        default=False,
        description="If True, rebuild even if index already exists",
    )


class BuildResponse(BaseModel):
    """Response body for /build endpoint."""
    message: str
    stats: dict


class StatusResponse(BaseModel):
    """Response body for /status endpoint."""
    status: str
    details: dict


# ────────────────────────────────────────────────────────────
# API ENDPOINTS
# ────────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a cricket question through the full RAG pipeline.
    Returns an answer with sources and metadata.
    """
    try:
        logger.info(f"📨 Question: {request.question[:80]}...")
        result = chatbot.ask(request.question)
        logger.info(
            f"✅ Answered in {result['processing_time']}s "
            f"({result['query_type']}, {result['search_results']} sources)"
        )
        return ChatResponse(**result)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error processing question: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your question: {str(e)}",
        )


@app.get("/status", response_model=StatusResponse)
async def status():
    """Get chatbot system status and statistics."""
    try:
        details = chatbot.get_status()
        return StatusResponse(status="ok", details=details)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build", response_model=BuildResponse)
async def build_index(request: BuildRequest):
    """
    Build or rebuild the FAISS + BM25 search index.
    This processes all cricket data files and creates embeddings.
    """
    try:
        logger.info(f"🔨 Building index (force_rebuild={request.force_rebuild})...")
        stats = chatbot.build_index(force_rebuild=request.force_rebuild)
        logger.info(f"✅ Index built: {stats}")
        return BuildResponse(
            message="Index built successfully",
            stats=stats,
        )
    except Exception as e:
        logger.error(f"Error building index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-history")
async def clear_history():
    """Clear the conversation history."""
    try:
        chatbot.clear_history()
        return {"message": "Conversation history cleared"}
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response via Server-Sent Events (SSE).
    Events: meta (query info), token (text chunk), done (final stats).
    """
    import asyncio

    def generate():
        try:
            for event_type, data in chatbot.ask_stream(request.question):
                yield f"event: {event_type}\ndata: {data}\n\n"
        except Exception as e:
            import json
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ────────────────────────────────────────────────────────────
# STATIC FILES (Frontend)
# ────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent / "Frontend"

if FRONTEND_DIR.exists():
    # Serve specific HTML pages
    @app.get("/")
    async def serve_index():
        """Serve the landing page."""
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/chat-page")
    async def serve_chatbot():
        """Serve the chatbot page."""
        return FileResponse(FRONTEND_DIR / "chatbot.html")

    # Mount static files (CSS, JS, images)
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("🏏 Launching Cricket World Cup RAG Server on http://localhost:8000")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
