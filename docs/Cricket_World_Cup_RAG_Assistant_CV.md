# Cricket World Cup RAG Assistant

## Project Overview

Developed a production-grade Retrieval-Augmented Generation (RAG) chatbot for ICC Cricket World Cup queries spanning 2003–2023 tournaments. The system leverages hybrid search techniques, AI-powered responses, and a comprehensive dataset of 299 matches to provide accurate, context-rich answers about cricket history, statistics, and memorable moments.

## Key Features

- **Hybrid Search Engine**: Implemented FAISS semantic search combined with BM25 keyword search and Reciprocal Rank Fusion (RRF) for optimal retrieval accuracy
- **AI-Powered Responses**: Integrated OpenRouter LLM with query-type-specific prompts and anti-hallucination safeguards
- **Intelligent Query Processing**: Built query classification system (statistical, comparative, match-specific, tournament, player) with entity resolution for player nicknames and team abbreviations
- **Multi-Interface Support**: Created both interactive CLI and polished FastAPI web server with HTML/CSS/JavaScript frontend
- **Rich Data Pipeline**: Processed and indexed 299 match files, tournament summaries, and player statistics using local sentence-transformers embeddings
- **Conversation Management**: Implemented conversation history, query rewriting, and context-aware responses

## Technical Implementation

### Backend Architecture
- **Language**: Python 3.8+
- **Framework**: FastAPI for REST API, async context management
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384D vectors, cosine similarity)
- **Vector Database**: FAISS HNSWFlat index with optimized search parameters
- **LLM Integration**: OpenRouter API with configurable models (Claude-3, GPT-4, etc.)
- **Data Processing**: Custom chunking strategy (200-400 tokens with overlap) for semantic coherence

### Data Engineering
- **Dataset Size**: 299 matches across 6 World Cup tournaments (2003, 2007, 2011, 2015, 2019, 2023)
- **Data Sources**: Curated match data, statistical analysis, player career aggregates
- **Preprocessing**: Entity normalization, alias resolution, temporal reference handling
- **Index Optimization**: SHA-256 deduplication, batch processing for 743 total vectors

### Frontend Development
- **Technologies**: HTML5, CSS3, Vanilla JavaScript
- **Design**: Responsive cricket-themed UI with smooth animations and transitions
- **Features**: Real-time chat interface, conversation history display, loading states
- **Styling**: Custom CSS with Oswald/Inter fonts, Lucide icons, mobile-first approach

### Key Algorithms & Techniques
- **Query Enhancement**: Multi-query generation for comprehensive retrieval
- **Context Re-ranking**: Score-based filtering and relevance optimization
- **Entity Resolution**: Player name aliases, team name variations
- **Prompt Engineering**: Type-specific system prompts for different query categories
- **Error Handling**: Graceful degradation, rate limiting, API key validation

## Skills Demonstrated

- **AI/ML**: RAG systems, vector embeddings, semantic search, LLM integration
- **Natural Language Processing**: Query classification, entity extraction, text chunking
- **Data Engineering**: Large dataset processing, vector indexing, search optimization
- **Web Development**: REST API design, frontend-backend integration, responsive UI
- **Software Architecture**: Modular design, configuration management, error handling
- **Python Ecosystem**: FastAPI, sentence-transformers, FAISS, OpenAI SDK
- **DevOps**: Environment configuration, logging, production deployment considerations

## Impact & Results

- **Accuracy**: Anti-hallucination measures ensure responses grounded in verified data
- **Performance**: Local embeddings eliminate API costs, sub-second query response times
- **Scalability**: Modular architecture supports additional tournaments and data sources
- **User Experience**: Intuitive interfaces for both technical and casual cricket fans
- **Maintainability**: Comprehensive documentation, type hints, and clean code structure

## Future Enhancements

- Real-time match data integration
- Multi-language support for global audience
- Advanced analytics dashboard
- Voice interaction capabilities
- Mobile app development

This project showcases expertise in building end-to-end AI applications that combine traditional software engineering with modern machine learning techniques, delivering valuable insights from complex datasets.