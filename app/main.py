"""
FastAPI application factory.
All app config, middleware, and route registration lives here.
Import `create_app()` from this module to build the application.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import agent, auth, documents, memory, rag
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.sqlite_store import init_database


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging()
    logger = logging.getLogger(__name__)

    app = FastAPI(
        title="PDF RAG Assistant",
        description="Production-ready PDF Retrieval-Augmented Generation API with multi-document support, hybrid search, ReAct agent, persistent memory, and JWT auth.",
        version="2.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router, tags=["Auth"])
    app.include_router(documents.router, tags=["Documents"])
    app.include_router(rag.router, tags=["RAG"])
    app.include_router(agent.router, tags=["Agent"])
    app.include_router(memory.router, tags=["Memory"])

    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting PDF RAG Assistant...")
        try:
            init_database()
            logger.info("SQLite storage initialized successfully")
        except Exception as exc:
            logger.warning("Could not initialize storage: %s", exc)

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "version": "2.1.0"}

    logger.info("PDF RAG Assistant app created")
    return app
