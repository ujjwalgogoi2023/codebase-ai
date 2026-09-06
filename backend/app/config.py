"""Application configuration, loaded from environment / .env.

Every knob the RAG pipeline exposes lives here rather than being scattered as
literals through the code (spec section 26). Two reasons that matters:

1. Secrets never end up in source. `.env` is gitignored; `.env.example` is the
   committed template.
2. The section 21 experiments need to vary chunk size, top-K and the embedding
   model WITHOUT editing code. If `top_k=10` is hardcoded in the retriever, an
   experiment becomes a code change, and results stop being comparable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed settings. Pydantic validates and coerces on load."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Provider credentials ---
    openai_api_key: str = ""

    # --- Database ---
    database_url: str = (
        "postgresql+psycopg://codebaseai:codebaseai@localhost:5433/codebaseai"
    )

    # --- Embeddings ---
    # "openai" for real vectors, "fake" for deterministic offline vectors.
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- Generation ---
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # --- Retrieval ---
    top_k: int = 10          # candidates pulled from the vector index
    rerank_top_k: int = 3    # survivors actually shown to the LLM

    # HNSW explores a wider slice of the graph as this rises: better recall,
    # slower queries. It is a real accuracy/latency dial, not a constant, so
    # the evaluation harness can sweep it.
    hnsw_ef_search: int = 64

    # --- Indexing limits ---
    max_file_bytes: int = 1_000_000
    max_chunk_lines: int = 120

    # --- Embedding request tuning ---
    embedding_batch_size: int = 128      # inputs per API request
    embedding_max_tokens: int = 8_000    # per input; model hard limit is 8191
    embedding_max_retries: int = 5

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    """Cached accessor.

    lru_cache means .env is parsed once per process, and FastAPI can use this
    directly as a dependency without re-reading the file on every request.
    """
    return Settings()
