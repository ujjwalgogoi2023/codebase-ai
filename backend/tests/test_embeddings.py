"""Tests for the embedding layer (RAG pipeline stage 3).

Nothing here touches the network or spends money. The OpenAI provider is
exercised against a fake client so the parts that actually break in production —
batching, result ordering, truncation, rate-limit retry — are all covered.
"""

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.services.chunker import chunk_file
from app.services.embeddings import (
    CachedEmbeddingProvider,
    EmbeddingError,
    FakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
    cosine_similarity,
    count_tokens,
    embed_chunks,
    estimate_cost,
    get_embedding_provider,
)
from app.services.repository_parser import SourceFile


# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------

def test_fake_provider_shape_and_determinism():
    provider = FakeEmbeddingProvider(dimension=64)

    a1 = provider.embed_one("def login(): pass")
    a2 = provider.embed_one("def login(): pass")
    b = provider.embed_one("def logout(): pass")

    assert len(a1) == 64
    assert a1 == a2, "same text must always give the same vector"
    assert a1 != b


def test_fake_provider_returns_unit_vectors():
    """Real providers return normalized vectors; the fake must match."""
    vector = FakeEmbeddingProvider(dimension=128).embed_one("x")
    magnitude = sum(v * v for v in vector) ** 0.5
    assert magnitude == pytest.approx(1.0, abs=1e-9)


def test_embed_preserves_order_and_count():
    provider = FakeEmbeddingProvider(dimension=32)
    texts = [f"chunk {i}" for i in range(10)]

    vectors = provider.embed(texts)

    assert len(vectors) == 10
    for text, vector in zip(texts, vectors):
        assert vector == provider.embed_one(text)


def test_embed_empty_list():
    assert FakeEmbeddingProvider().embed([]) == []


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def test_cosine_similarity_reference_cases():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)
    # Magnitude must not matter, only direction.
    assert cosine_similarity([1, 1], [5, 5]) == pytest.approx(1.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


# ---------------------------------------------------------------------------
# OpenAI provider, against a fake client
# ---------------------------------------------------------------------------

class FakeOpenAIClient:
    """Records calls and returns deterministic vectors."""

    def __init__(self, dimension: int = 3, shuffle: bool = False, fail_times: int = 0):
        self.calls: list[list[str]] = []
        self.dimension = dimension
        self.shuffle = shuffle
        self.fail_times = fail_times
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input):  # noqa: A002 - mirrors the real signature
        if self.fail_times > 0:
            self.fail_times -= 1
            raise _rate_limit_error()

        self.calls.append(list(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(i)] * self.dimension)
            for i in range(len(input))
        ]
        if self.shuffle:
            # The API documents an `index` field because arrival order is not
            # guaranteed. Reverse it to prove we sort rather than trust order.
            data = list(reversed(data))
        return SimpleNamespace(data=data)


def _rate_limit_error():
    import httpx
    from openai import RateLimitError

    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(429, request=request)
    return RateLimitError("slow down", response=response, body=None)


def make_openai_provider(client: FakeOpenAIClient, **kwargs) -> OpenAIEmbeddingProvider:
    provider = OpenAIEmbeddingProvider(api_key="sk-test", dimension=3, **kwargs)
    provider._client = client
    return provider


def test_openai_provider_batches_requests():
    client = FakeOpenAIClient()
    provider = make_openai_provider(client, batch_size=4)

    vectors = provider.embed([f"text {i}" for i in range(10)])

    assert len(vectors) == 10
    assert [len(c) for c in client.calls] == [4, 4, 2], "did not batch as configured"


def test_openai_provider_sorts_results_by_index():
    """A mis-zip here would attach every vector to the wrong chunk."""
    client = FakeOpenAIClient(shuffle=True)
    provider = make_openai_provider(client, batch_size=10)

    vectors = provider.embed(["a", "b", "c"])

    assert vectors == [[0.0] * 3, [1.0] * 3, [2.0] * 3]


def test_openai_provider_truncates_oversized_input():
    client = FakeOpenAIClient()
    provider = make_openai_provider(client, max_input_tokens=5)

    provider.embed(["word " * 500])

    sent = client.calls[0][0]
    assert count_tokens([sent]) <= 5


def test_openai_provider_retries_rate_limits(monkeypatch):
    import app.services.embeddings as embeddings_module

    slept: list[int] = []
    monkeypatch.setattr(embeddings_module, "_sleep_backoff", lambda a: slept.append(a))

    client = FakeOpenAIClient(fail_times=2)
    provider = make_openai_provider(client, max_retries=5)

    vectors = provider.embed(["a"])

    assert len(vectors) == 1
    assert slept == [0, 1], "should back off once per failed attempt"


def test_openai_provider_gives_up_after_max_retries(monkeypatch):
    import app.services.embeddings as embeddings_module
    from openai import RateLimitError

    monkeypatch.setattr(embeddings_module, "_sleep_backoff", lambda a: None)
    provider = make_openai_provider(FakeOpenAIClient(fail_times=99), max_retries=3)

    with pytest.raises(RateLimitError):
        provider.embed(["a"])


def test_openai_provider_requires_a_key():
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY is empty"):
        OpenAIEmbeddingProvider(api_key="   ")


def test_openai_provider_empty_input_makes_no_call():
    client = FakeOpenAIClient()
    provider = make_openai_provider(client)

    assert provider.embed([]) == []
    assert client.calls == []


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class CountingProvider(FakeEmbeddingProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embed_calls: list[list[str]] = []

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        return super().embed(texts)


def test_cache_avoids_repeat_calls(tmp_path):
    inner = CountingProvider(dimension=16)
    cache = CachedEmbeddingProvider(inner, cache_path=tmp_path / "c.sqlite")

    first = cache.embed(["a", "b"])
    second = cache.embed(["a", "b"])

    assert first == second
    assert cache.hits == 2 and cache.misses == 2
    assert len(inner.embed_calls) == 1, "second call should be served from cache"
    cache.close()


def test_cache_only_fetches_the_missing_texts(tmp_path):
    inner = CountingProvider(dimension=16)
    cache = CachedEmbeddingProvider(inner, cache_path=tmp_path / "c.sqlite")

    cache.embed(["a"])
    result = cache.embed(["a", "b", "c"])

    assert len(result) == 3
    assert inner.embed_calls[-1] == ["b", "c"], "cached text was re-fetched"
    cache.close()


def test_cache_survives_reopen(tmp_path):
    path = tmp_path / "c.sqlite"
    first = CachedEmbeddingProvider(CountingProvider(dimension=16), cache_path=path)
    vectors = first.embed(["persisted"])
    first.close()

    inner = CountingProvider(dimension=16)
    second = CachedEmbeddingProvider(inner, cache_path=path)
    assert second.embed(["persisted"]) == vectors
    assert inner.embed_calls == [], "should not have called the provider at all"
    second.close()


def test_cache_key_includes_the_model(tmp_path):
    """Switching models must never serve vectors from the old one."""
    path = tmp_path / "c.sqlite"

    a = CachedEmbeddingProvider(FakeEmbeddingProvider(dimension=16, model="m1"), cache_path=path)
    va = a.embed(["same text"])
    a.close()

    inner = CountingProvider(dimension=16, model="m2")
    b = CachedEmbeddingProvider(inner, cache_path=path)
    vb = b.embed(["same text"])
    b.close()

    assert inner.embed_calls == [["same text"]], "stale vector served across models"
    assert va != vb


# ---------------------------------------------------------------------------
# Factory + config
# ---------------------------------------------------------------------------

def test_factory_builds_fake_provider():
    settings = Settings(embedding_provider="fake", embedding_dim=256)
    provider = get_embedding_provider(settings)

    assert isinstance(provider, FakeEmbeddingProvider)
    assert provider.dimension == 256


def test_factory_wraps_openai_in_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.embeddings.PROJECT_ROOT", tmp_path
    )
    settings = Settings(embedding_provider="openai", openai_api_key="sk-test")

    provider = get_embedding_provider(settings)
    assert isinstance(provider, CachedEmbeddingProvider)
    assert isinstance(provider.inner, OpenAIEmbeddingProvider)
    provider.close()


def test_factory_builds_ollama_provider_without_a_key(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.embeddings.PROJECT_ROOT", tmp_path)
    settings = Settings(
        embedding_provider="ollama",
        openai_api_key="",
        embedding_model="qwen3-embedding:0.6b",
        embedding_dim=1024,
        ollama_base_url="http://localhost:11434/v1",
    )

    provider = get_embedding_provider(settings)
    assert isinstance(provider, CachedEmbeddingProvider)
    assert isinstance(provider.inner, OpenAIEmbeddingProvider)
    assert str(provider.inner._client.base_url).startswith("http://localhost:11434/v1")
    assert provider.dimension == 1024
    provider.close()


def test_openai_provider_rejects_a_dimension_mismatch():
    # The model returns 3-d vectors but the schema was built for 5.
    provider = make_openai_provider(FakeOpenAIClient(dimension=3))
    provider.dimension = 5
    with pytest.raises(EmbeddingError, match="EMBEDDING_DIM=3"):
        provider.embed(["x"])


def test_openai_provider_explains_an_unreachable_server(monkeypatch):
    import httpx
    from openai import APIConnectionError

    monkeypatch.setattr("app.services.embeddings._sleep_backoff", lambda attempt: None)

    def refuse(model, input):  # noqa: A002
        raise APIConnectionError(request=httpx.Request("POST", "http://localhost:11434"))

    provider = OpenAIEmbeddingProvider(
        api_key="", base_url="http://localhost:11434/v1", dimension=3, max_retries=2
    )
    provider._client.embeddings = SimpleNamespace(create=refuse)
    with pytest.raises(EmbeddingError, match="ollama serve"):
        provider.embed(["x"])


def test_factory_rejects_unknown_provider():
    settings = Settings(embedding_provider="pinecone-magic")
    with pytest.raises(EmbeddingError, match="Unknown embedding provider"):
        get_embedding_provider(settings)


def test_factory_reports_a_missing_key_clearly():
    settings = Settings(embedding_provider="openai", openai_api_key="")
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY is empty"):
        get_embedding_provider(settings)


# ---------------------------------------------------------------------------
# Chunk integration + cost
# ---------------------------------------------------------------------------

def test_embed_chunks_uses_path_enriched_text():
    source = SourceFile(
        path="backend/auth.py",
        language="python",
        content="def login():\n    return 1\n",
        num_lines=2,
        size_bytes=26,
    )
    chunks = chunk_file(source, min_chunk_lines=1)

    class Recorder(FakeEmbeddingProvider):
        seen: list[str] = []

        def embed(self, texts):
            Recorder.seen = list(texts)
            return super().embed(texts)

    vectors = embed_chunks(chunks, Recorder(dimension=8))

    assert len(vectors) == len(chunks)
    assert any("backend/auth.py" in text for text in Recorder.seen), (
        "file path must reach the embedding model, not just the metadata"
    )


def test_estimate_cost():
    estimate = estimate_cost(["hello world"] * 100, model="text-embedding-3-small")

    assert estimate["texts"] == 100
    assert estimate["tokens"] > 0
    assert estimate["usd"] >= 0.0


def test_estimate_cost_unknown_model_reports_none():
    assert estimate_cost(["x"], model="some-future-model")["usd"] is None


# ---------------------------------------------------------------------------
# Quota vs rate limit: a 429 is two different failures wearing one status code
# ---------------------------------------------------------------------------

def _quota_error():
    import httpx
    from openai import RateLimitError

    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    body = {"error": {"message": "You have no credits remaining.",
                      "type": "insufficient_quota",
                      "code": "credit_balance_exhausted"}}
    response = httpx.Response(429, request=request, json=body)
    return RateLimitError("You have no credits remaining.", response=response, body=body)


class QuotaFailingClient(FakeOpenAIClient):
    def _create(self, model, input):  # noqa: A002
        self.calls.append(list(input))
        raise _quota_error()


def test_quota_error_fails_fast_without_retrying(monkeypatch):
    """Retrying 'out of credits' just wastes time to reach the same failure."""
    import app.services.embeddings as embeddings_module

    slept: list[int] = []
    monkeypatch.setattr(embeddings_module, "_sleep_backoff", lambda a: slept.append(a))

    client = QuotaFailingClient()
    provider = make_openai_provider(client, max_retries=5)

    with pytest.raises(EmbeddingError, match="no credits remaining"):
        provider.embed(["a"])

    assert slept == [], "must not back off on a permanent billing error"
    assert len(client.calls) == 1, "must not retry a quota failure"


def test_transient_rate_limit_still_retries(monkeypatch):
    """The fast-fail must not swallow ordinary rate limiting."""
    import app.services.embeddings as embeddings_module

    monkeypatch.setattr(embeddings_module, "_sleep_backoff", lambda a: None)
    client = FakeOpenAIClient(fail_times=1)
    provider = make_openai_provider(client, max_retries=3)

    assert len(provider.embed(["a"])) == 1
