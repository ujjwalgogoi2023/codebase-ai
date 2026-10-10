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

    # Ollama serves an OpenAI-compatible API, so the "ollama" provider reuses
    # the OpenAI client pointed here. Local, free, and needs no key.
    ollama_base_url: str = "http://localhost:11434/v1"

    # --- Database ---
    database_url: str = (
        "postgresql+psycopg://codebaseai:codebaseai@localhost:5433/codebaseai"
    )

    # --- Embeddings ---
    # "openai" or "ollama" for real vectors, "fake" for deterministic offline
    # vectors.
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- Generation ---
    # "openai" or "ollama" for real answers, "fake" for scripted offline
    # responses.
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # Zero temperature on purpose. A grounded answer should be reproducible:
    # the same question over the same code should give the same answer, or the
    # evaluation numbers in section 20 measure sampling noise as much as
    # retrieval quality.
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 1_000
    llm_max_retries: int = 4

    # Budget for retrieved code in the prompt. Context is not free: it costs
    # money, adds latency, and past a point actively hurts, because the useful
    # chunk gets buried among mediocre ones.
    max_context_tokens: int = 8_000

    # How many prior turns of a conversation to replay. Enough for "how does it
    # validate the token?" to make sense; not so many that old turns dominate.
    max_history_turns: int = 6

    # --- Retrieval ---
    top_k: int = 10          # candidates pulled from the vector index
    rerank_top_k: int = 3    # survivors actually shown to the LLM

    # HNSW explores a wider slice of the graph as this rises: better recall,
    # slower queries. It is a real accuracy/latency dial, not a constant, so
    # the evaluation harness can sweep it.
    hnsw_ef_search: int = 64

    # --- Reranking ---
    # "llm" re-scores candidates with the language model; "identity" keeps the
    # vector order and is the control group for the section 21 experiment.
    reranker: str = "llm"
    # Lines of each candidate shown to the ranker. A signature, a docstring and
    # the first few statements carry nearly all the signal about what code
    # does; sending whole chunks would make ranking cost as much as answering.
    rerank_preview_lines: int = 18

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
