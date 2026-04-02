## PDF RAG Backend

FastAPI backend for the PDF RAG platform. This service handles authentication, PDF upload and processing, retrieval, agent responses, memory management, usage tracking, and background jobs.

## Overview

This repo provides:

- JWT-based authentication
- per-user PDF upload and storage
- asynchronous PDF processing with Redis and RQ
- hybrid retrieval using embeddings and keyword search
- ReAct-style agent responses with tools and reasoning steps
- memory and chat-history management
- SQLite-backed persistence for users, chunks, memories, and usage
- JSON and streaming API responses

The companion frontend repo is `pdf-rag-app`.

## Key Features

- FastAPI app with Swagger docs at `/docs`
- per-user document isolation
- background ingestion pipeline for uploads
- hybrid search over persisted chunks
- streaming endpoints for RAG and agent answers
- daily usage tracking and request limits
- Docker Compose setup for API, worker, and Redis

## Architecture

### Request flow

1. A client authenticates with `/register` or `/login`.
2. A PDF is uploaded to `/upload`.
3. The API stores the file and enqueues a background job.
4. The worker processes the document, creates chunks and embeddings, and persists them.
5. Query endpoints retrieve relevant context from the per-user document store.
6. The RAG service or agent generates an answer and returns JSON or SSE events.

### Application layers

- `app/api/routes`: HTTP endpoints
- `app/services`: route-facing service layer
- `app/services/internals`: retrieval, agent, prompts, chunking, memory internals
- `app/db`: SQLite and vector-store logic
- `app/core`: config, auth, logging
- `app/models`: request and response schemas
- `app/tools`: agent tool registry and schemas

## Repository Structure

```text
pdf-rag-backend/
|- app/
|  |- api/routes/         # auth, documents, rag, agent, memory endpoints
|  |- core/               # configuration, auth, logging
|  |- db/                 # SQLite and vector-store persistence
|  |- models/             # Pydantic schemas
|  |- services/           # service facades and job orchestration
|  |- services/internals/ # retrieval, agent, memory, prompt logic
|  |- tools/              # agent tool definitions
|  |- main.py             # FastAPI application factory
|  |- queue.py            # Redis/RQ helpers
|- main.py                # process entrypoint
|- worker.py              # RQ worker entrypoint
|- compose.yaml           # local multi-service runtime
|- Dockerfile             # production image
|- test_*.py              # tests
```

## Tech Stack

- Python 3.13
- FastAPI
- Uvicorn and Gunicorn
- OpenAI SDK
- FAISS
- SQLite
- Redis + RQ
- PyPDF

## Configuration

Copy `.env.example` to `.env`.

### Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | Yes | - | OpenAI API access |
| `MODEL_NAME` | No | `gpt-4o-mini` | LLM used for generation |
| `UPLOAD_DIR` | No | `uploaded/` | Uploaded files directory |
| `CACHE_DIR` | No | `cache/` | Cache directory |
| `PERSISTENCE_DIR` | No | `persistence/` | Persistence root |
| `SQLITE_DB_PATH` | No | `persistence/app.db` | SQLite database file |
| `JWT_SECRET_KEY` | Yes for production | `change-me-in-production` | JWT signing secret |
| `JWT_ALGORITHM` | No | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `1440` | Access token lifetime |
| `REDIS_URL` | Yes for uploads | `redis://localhost:6379` | Redis connection string |
| `RQ_QUEUE_NAME` | No | `default` | RQ queue name |

## Local Development

### Recommended: Docker Compose

```bash
docker compose up --build
```

This starts:

- API on `http://localhost:8000`
- Swagger docs on `http://localhost:8000/docs`
- worker process for upload jobs
- Redis on `localhost:6379`

### Run without Docker

Install dependencies:

```bash
uv sync
```

Run the API:

```bash
uv run uvicorn main:app --reload
```

Run the worker in another terminal:

```bash
uv run python worker.py
```

Make sure Redis is running before testing uploads.

### Windows note

`app/queue.py` guards against RQ import issues on native Windows. If the worker fails locally, use Docker, WSL, or Linux for the API/worker runtime.

## API Overview

### Auth

- `POST /register`
- `POST /login`
- `GET /me`

### Documents

- `POST /upload`
- `GET /documents`
- `DELETE /documents`
- `GET /job/{job_id}`
- `GET /status`

### RAG

- `POST /ask`
- `POST /ask-stream`

### Agent

- `POST /agent`
- `POST /agent-stream`

### Memory

- `GET /memory/stats`
- `GET /memory/info`
- `DELETE /memory/chat`
- `DELETE /memory/all`
- `POST /memory/cleanup`

### Health

- `GET /health`

## Data and Persistence

- SQLite stores users, document chunks, memories, chat history, and usage
- uploaded files are written under `UPLOAD_DIR/<username>/`
- chunk embeddings are persisted in SQLite and loaded into per-user FAISS indexes
- Redis stores queue and job state for asynchronous ingestion

## Testing

Run the full test suite:

```bash
uv run pytest
```

Examples of focused tests already in the repo:

- `test_documents_route.py`
- `test_upload_background_task.py`
- `test_auth_tokens.py`
- `test_usage_tracking.py`

## Deployment

### Docker

The `Dockerfile` runs the API with Gunicorn and Uvicorn workers.

### Railway

This repo also includes:

- `railway.toml`
- `railway.env.example`
- `RAILWAY_DEPLOYMENT.md`

Before deploying to production:

- set a real `JWT_SECRET_KEY`
- tighten CORS in `app/core/config.py`
- provision Redis for upload jobs
- persist upload and SQLite storage

## Related Repository

- Frontend: `C:\development\pdf-rag-app`
