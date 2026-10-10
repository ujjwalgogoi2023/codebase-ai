"""Embeddings: turn code chunks into vectors.

Stage 3 of the RAG ingestion pipeline:

    Repository -> Parsing -> Chunking -> [EMBEDDINGS] -> pgvector

WHAT AN EMBEDDING IS
--------------------
An embedding model maps text to a fixed-length list of floats:

    def authenticate_user(...)  ->  [0.013, -0.234, 0.881, ...]   (1536 numbers)

The model is trained so that text with similar MEANING lands near the same
direction in that space. That is the entire trick behind RAG: it lets us find
"the function that checks a user's password" without the query sharing a single
keyword with the code. Keyword search cannot do that; `grep "authenticate"`
misses `verify_credentials()` entirely.

HOW VECTORS ARE COMPARED
------------------------
By the angle between them, not the distance:

    cosine_similarity(a, b) = (a . b) / (|a| * |b|)

Angle rather than length, because length mostly encodes how much text there was,
not what it meant. A 3-line function and a 100-line function about the same
topic should still match.

Useful detail: OpenAI returns unit-length vectors (|v| = 1), so cosine
similarity collapses to a plain dot product. That is why pgvector's `<=>`
cosine operator is cheap for us later.

WHY A PROVIDER ABSTRACTION
--------------------------
Two concrete reasons, not architecture astronomy:

1. Tests must never spend money or need network. FakeEmbeddingProvider produces
   deterministic vectors offline, so the whole retrieval pipeline is testable.
2. Swapping the embedding model is a real experiment (section 21). That should
   be an .env change, not a refactor.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import PROJECT_ROOT, Settings, get_settings

# Cost per 1M tokens, USD. Used only for the pre-flight estimate, so you are
# never surprised by an indexing bill.
EMBEDDING_PRICES_PER_1M_TOKENS: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

KNOWN_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    # Local models served by Ollama.
    "qwen3-embedding:0.6b": 1024,
    "nomic-embed-text": 768,
    "bge-m3": 1024,
}


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Anything that can turn text into vectors."""

    model: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, returning one vector per input, in the same order.

        Order matters enormously: we zip these back onto chunks by position. A
        provider that reorders results would silently attach every vector to
        the wrong chunk, and retrieval would return confident nonsense.
        """

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# Fake provider: offline, deterministic, free
# ---------------------------------------------------------------------------

class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic pseudo-embeddings derived from a hash of the text.

    These carry no real semantics — they cannot tell "login" from "logout". They
    exist so the storage, retrieval and API layers can be tested end to end
    without network or spend. Any test that asserts on SEMANTIC quality must use
    a real provider; tests using this one may only assert on plumbing.
    """

    def __init__(self, dimension: int = 1536, model: str = "fake-embedding") -> None:
        self.dimension = dimension
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        # The model name is part of the seed: two different models must not
        # produce identical vectors, or tests that switch models would silently
        # compare a model against itself.
        payload = f"{self.model}\x00{text}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        rng = random.Random(seed)
        raw = [rng.gauss(0.0, 1.0) for _ in range(self.dimension)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        # Unit length, matching what real providers return.
        return [v / norm for v in raw]


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Real embeddings via the OpenAI API, or any server that speaks it.

    Handles the three things that break naive implementations at scale:
    batching, per-input token limits, and rate limits.

    With `base_url` set it talks to an OpenAI-compatible server instead, which
    is how the "ollama" provider runs local models for free. No second client
    to write or test: the wire format is identical.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int | None = None,
        batch_size: int = 128,
        max_input_tokens: int = 8_000,
        max_retries: int = 5,
        base_url: str | None = None,
    ) -> None:
        if not api_key.strip() and base_url is None:
            raise EmbeddingError(
                "OPENAI_API_KEY is empty. Add it to .env, or set "
                "EMBEDDING_PROVIDER=fake to run offline."
            )

        from openai import OpenAI  # imported lazily so offline tests need no SDK

        self.model = model
        self.dimension = dimension or KNOWN_DIMENSIONS.get(model, 1536)
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.max_retries = max_retries
        # The SDK retries transport errors itself; we handle rate limits below.
        self._client = OpenAI(
            api_key=api_key or "unused", base_url=base_url, max_retries=2
        )
        self._encoder = _get_encoder(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        prepared = [self._truncate(t) for t in texts]
        vectors: list[list[float]] = []

        for batch in _batched(prepared, self.batch_size):
            vectors.extend(self._embed_batch(batch))

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Provider returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        # Caught here rather than at INSERT, where pgvector reports it as an
        # opaque "expected 1536 dimensions, not 1024" deep inside indexing.
        if len(vectors[0]) != self.dimension:
            raise EmbeddingError(
                f"{self.model} returns {len(vectors[0])}-dimensional vectors but "
                f"EMBEDDING_DIM={self.dimension}. Set EMBEDDING_DIM="
                f"{len(vectors[0])} in .env and recreate the schema "
                "(python scripts/init_db.py --reset)."
            )
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        from openai import APIConnectionError, APIError, RateLimitError

        for attempt in range(self.max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self.model, input=batch
                )
                # Sort by index rather than trusting arrival order — the API
                # documents an `index` field precisely because order is not
                # guaranteed, and a silent mis-zip here is unrecoverable.
                items = sorted(response.data, key=lambda d: d.index)
                return [item.embedding for item in items]

            except RateLimitError as exc:
                # HTTP 429 covers two very different situations. "Too many
                # requests" is transient and worth backing off for. "You have
                # no credits" is permanent — retrying it just wastes 30
                # seconds and hammers the API to reach the same failure.
                if _is_quota_error(exc):
                    raise EmbeddingError(
                        "OpenAI rejected the request: the account has no "
                        "credits remaining. Add credits at "
                        "https://platform.openai.com/settings/organization/billing/ "
                        "or set EMBEDDING_PROVIDER=fake in .env to work offline."
                    ) from exc
                if attempt == self.max_retries - 1:
                    raise
                _sleep_backoff(attempt)

            except APIConnectionError as exc:
                # Usually a local server that is not running, rather than a
                # network blip, so say where we were trying to connect.
                if attempt == self.max_retries - 1:
                    raise EmbeddingError(
                        f"Could not reach {self._client.base_url}. If this is "
                        "Ollama, start it (ollama serve) and pull the model "
                        f"(ollama pull {self.model})."
                    ) from exc
                _sleep_backoff(attempt)

            except APIError as exc:
                # 5xx is worth retrying; a 4xx is our bug and should surface now.
                status = getattr(exc, "status_code", None)
                if status is not None and status < 500:
                    raise
                if attempt == self.max_retries - 1:
                    raise
                _sleep_backoff(attempt)

        raise EmbeddingError("exhausted retries")  # pragma: no cover

    def _truncate(self, text: str) -> str:
        """Clip to the model's input limit.

        A single oversized input fails the WHOLE batch, so one 9000-token
        vendored file would otherwise take 127 healthy chunks down with it.
        Truncating is the pragmatic choice: the head of a chunk carries the
        signature and docstring, which is most of its retrievable meaning.
        """
        if self._encoder is None:
            # Rough fallback: ~4 characters per token.
            limit = self.max_input_tokens * 4
            return text[:limit]

        tokens = self._encoder.encode(text)
        if len(tokens) <= self.max_input_tokens:
            return text
        return self._encoder.decode(tokens[: self.max_input_tokens])


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------

class CachedEmbeddingProvider(EmbeddingProvider):
    """Wraps a provider with a local SQLite cache keyed by (model, text).

    Re-indexing the same repository is common during development — you restart,
    you tweak the retriever, you re-run the evaluation. Without a cache you pay
    for identical vectors every time. With it, the second run is free and
    instant.

    The key includes the model name, so switching models never serves stale
    vectors from the previous one.
    """

    def __init__(self, inner: EmbeddingProvider, cache_path: Path | None = None) -> None:
        self.inner = inner
        self.model = inner.model
        self.dimension = inner.dimension
        self.hits = 0
        self.misses = 0

        self.cache_path = cache_path or (PROJECT_ROOT / "data" / "embedding_cache.sqlite")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.cache_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  key TEXT PRIMARY KEY,"
            "  vector TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self.model}\x00{text}".encode("utf-8")).hexdigest()
        return digest

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        keys = [self._key(t) for t in texts]
        found: dict[str, list[float]] = {}

        # SQLite caps variables per statement, so look up in windows.
        for window in _batched(list(set(keys)), 400):
            placeholders = ",".join("?" * len(window))
            rows = self._conn.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",
                window,
            ).fetchall()
            for key, blob in rows:
                found[key] = json.loads(blob)

        missing_idx = [i for i, key in enumerate(keys) if key not in found]
        self.hits += len(texts) - len(missing_idx)
        self.misses += len(missing_idx)

        if missing_idx:
            fresh = self.inner.embed([texts[i] for i in missing_idx])
            to_store = []
            for i, vector in zip(missing_idx, fresh):
                found[keys[i]] = vector
                to_store.append((keys[i], json.dumps(vector)))
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                to_store,
            )
            self._conn.commit()

        return [found[key] for key in keys]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batched(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# Codes and phrases that mean "this 429 will never succeed on retry".
_QUOTA_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "billing_hard_limit_reached",
    "exceeded your current quota",
    "no credits remaining",
)


def _is_quota_error(exc: Exception) -> bool:
    """Distinguish a permanent billing 429 from a transient rate-limit 429."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code in _QUOTA_MARKERS:
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") or {}
        if isinstance(error, dict):
            for field in ("code", "type"):
                value = error.get(field)
                if isinstance(value, str) and value in _QUOTA_MARKERS:
                    return True

    message = str(exc).lower()
    return any(marker in message for marker in _QUOTA_MARKERS)


def _sleep_backoff(attempt: int) -> None:
    """Exponential backoff with jitter: 1s, 2s, 4s, 8s (+/- randomness).

    Jitter matters because without it, every worker that got rate-limited at the
    same moment retries at the same moment, and the burst repeats forever.
    """
    delay = (2 ** attempt) * (0.5 + random.random())
    time.sleep(min(delay, 30.0))


def _get_encoder(model: str):
    """tiktoken encoder for token counting, or None if unavailable."""
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - tiktoken is a hard dependency
        return None


def count_tokens(texts: list[str], model: str = "text-embedding-3-small") -> int:
    encoder = _get_encoder(model)
    if encoder is None:
        return sum(len(t) for t in texts) // 4
    return sum(len(encoder.encode(t)) for t in texts)


def estimate_cost(texts: list[str], model: str = "text-embedding-3-small") -> dict:
    """Pre-flight cost estimate, so indexing is never a surprise charge."""
    tokens = count_tokens(texts, model)
    price = EMBEDDING_PRICES_PER_1M_TOKENS.get(model)
    return {
        "texts": len(texts),
        "tokens": tokens,
        "model": model,
        "usd": round(tokens / 1_000_000 * price, 4) if price else None,
    }


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Reference implementation.

    pgvector does this in the database for real queries; this exists so the
    concept is visible in the codebase and testable without a database.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedding_provider(
    settings: Settings | None = None,
    *,
    use_cache: bool = True,
) -> EmbeddingProvider:
    """Build the provider named in configuration."""
    settings = settings or get_settings()
    provider_name = settings.embedding_provider.strip().lower()

    if provider_name == "fake":
        return FakeEmbeddingProvider(dimension=settings.embedding_dim)

    if provider_name == "openai":
        provider: EmbeddingProvider = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            max_input_tokens=settings.embedding_max_tokens,
            max_retries=settings.embedding_max_retries,
        )
        return CachedEmbeddingProvider(provider) if use_cache else provider

    if provider_name == "ollama":
        provider = OpenAIEmbeddingProvider(
            api_key="",
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            max_input_tokens=settings.embedding_max_tokens,
            max_retries=settings.embedding_max_retries,
        )
        return CachedEmbeddingProvider(provider) if use_cache else provider

    raise EmbeddingError(
        f"Unknown embedding provider {settings.embedding_provider!r}. "
        "Supported: 'openai', 'ollama', 'fake'."
    )


def embed_chunks(chunks, provider: EmbeddingProvider) -> list[list[float]]:
    """Embed CodeChunks, using each chunk's path/symbol-enriched text.

    Deliberately returns vectors positionally rather than mutating the chunks:
    CodeChunk is frozen, and keeping vectors separate makes the zip explicit at
    the call site where storage happens.
    """
    return provider.embed([chunk.embedding_text for chunk in chunks])
