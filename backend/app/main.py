"""FastAPI application entry point.

Run it with:

    uvicorn app.main:app --reload --app-dir backend

Interactive docs are at /docs, which FastAPI generates from the Pydantic
schemas. That is most of the reason FastAPI is the right choice here: the
response models are already written for validation, and documentation falls
out of them for free.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api import chat, repositories
from app.config import get_settings
from app.db.base import get_engine, init_db
from app.services.embeddings import get_embedding_provider
from app.services.llm import get_llm_provider
from app.schemas import HealthOut

DESCRIPTION = """\
Ask natural-language questions about a codebase and get answers grounded in the
actual source, with file and line citations.

Pipeline: parse -> code-aware chunk -> embed -> pgvector -> retrieve -> rerank
-> LLM -> answer with citations.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Creating the schema on startup keeps `docker compose up` to a single
    # step. It is create-if-missing, so it is safe to run every boot; a
    # production deployment would use migrations instead.
    init_db()
    yield


app = FastAPI(
    title="CodebaseAI",
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
)

# The React dev server runs on a different origin, so the browser needs
# permission to call this API. Origins are listed explicitly rather than "*",
# which is the habit worth keeping even when nothing here is authenticated yet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(repositories.router)
app.include_router(chat.router)


@app.get("/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    """Liveness plus the configuration actually in effect.

    Reporting the providers is deliberate: "why are my answers nonsense?" is
    very often "the server is running with EMBEDDING_PROVIDER=fake", and that
    should take one request to discover rather than an afternoon.
    """
    settings = get_settings()

    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        database = "ok"
    except Exception as exc:
        database = f"error: {type(exc).__name__}"

    # Report the model each provider will ACTUALLY use, not the configured
    # string. With EMBEDDING_PROVIDER=fake, `embedding_model` still reads
    # "text-embedding-3-small" from .env while the running provider is
    # "fake-embedding" -- which is precisely the confusion this endpoint is
    # supposed to remove.
    return HealthOut(
        status="ok" if database == "ok" else "degraded",
        database=database,
        embedding_provider=settings.embedding_provider,
        embedding_model=_active_model(get_embedding_provider, settings),
        llm_provider=settings.llm_provider,
        llm_model=_active_model(get_llm_provider, settings),
    )


def _active_model(factory, settings) -> str:
    """The live provider's model name, or why it could not be built.

    A provider can legitimately fail to construct (a missing API key, for
    instance). Health must still answer in that case, since "the key is
    missing" is exactly the diagnosis someone is here for.
    """
    try:
        return factory(settings).model
    except Exception as exc:
        return f"unavailable: {exc.__class__.__name__}"
