"""Tests for the FastAPI layer (pipeline stage 8).

Every dependency is overridden to point at the test database and the offline
providers, so the whole HTTP surface is exercised without network or spend.

TestClient runs background tasks synchronously once the response is sent, so a
POST that schedules indexing has finished indexing by the time the assertion
runs. Real clients poll /status instead, which is what the status tests cover.
"""

from __future__ import annotations

import io
import zipfile
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.config import Settings
from app.db.base import get_session_factory
from app.main import app
from app.models import Conversation, Message
from app.services.embeddings import FakeEmbeddingProvider
from app.services.indexer import index_repository
from app.services.llm import FakeLLMProvider

pytestmark = pytest.mark.usefixtures("_schema")

ANSWER = "Authentication happens in the login handler [1]."


@pytest.fixture
def llm() -> FakeLLMProvider:
    return FakeLLMProvider(response=ANSWER)


@pytest.fixture
def client(db, database_url, llm):
    settings = Settings(
        embedding_provider="fake",
        llm_provider="fake",
        embedding_dim=1536,
        top_k=5,
    )
    embeddings = FakeEmbeddingProvider(dimension=1536, model="fake-embedding")

    @contextmanager
    def background_session():
        """A separate session, as a real background task would need."""
        session = get_session_factory(database_url)()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[deps.get_db] = lambda: db
    app.dependency_overrides[deps.get_config] = lambda: settings
    app.dependency_overrides[deps.get_embeddings] = lambda: embeddings
    app.dependency_overrides[deps.get_llm] = lambda: llm
    app.dependency_overrides[deps.get_background_session] = lambda: background_session

    # Constructed without `with`, so the lifespan hook (which would call
    # init_db against the DEVELOPMENT database) never runs.
    yield TestClient(app)

    app.dependency_overrides.clear()


@pytest.fixture
def indexed(db, sample_repo):
    return index_repository(
        db,
        name="demo",
        root=sample_repo,
        provider=FakeEmbeddingProvider(dimension=1536, model="fake-embedding"),
    )


def make_zip_bytes(entries: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_reports_database_and_providers(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["database"] == "ok"
    # Surfacing this is the point: "why are answers nonsense" is often "the
    # server is running on fakes".
    assert body["embedding_provider"] in ("fake", "openai")


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "CodebaseAI"
    assert "/chat" in schema["paths"]
    assert "/repositories/{repository_id}/search" in schema["paths"]


# ---------------------------------------------------------------------------
# Registering and indexing
# ---------------------------------------------------------------------------

def test_register_local_path_indexes_in_background(client, sample_repo):
    response = client.post(
        "/repositories", json={"path": str(sample_repo), "name": "demo"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "demo"
    assert body["source_type"] == "local"

    # The background task has run by now under TestClient.
    status_body = client.get(f"/repositories/{body['id']}/status").json()
    assert status_body["index_status"] == "ready"
    assert status_body["file_count"] == 4
    assert status_body["chunk_count"] > 0


def test_register_returns_before_indexing_completes(client, sample_repo):
    """The POST response itself must not wait for indexing."""
    body = client.post(
        "/repositories", json={"path": str(sample_repo), "index_now": False}
    ).json()

    assert body["index_status"] == "pending"
    assert body["chunk_count"] == 0


def test_name_defaults_to_the_folder_name(client, sample_repo):
    body = client.post("/repositories", json={"path": str(sample_repo)}).json()
    assert body["name"] == sample_repo.resolve().name


def test_register_requires_exactly_one_source(client, sample_repo):
    assert client.post("/repositories", json={}).status_code == 422
    both = client.post(
        "/repositories",
        json={"path": str(sample_repo), "url": "https://github.com/a/b"},
    )
    assert both.status_code == 422


def test_register_rejects_a_missing_directory(client):
    response = client.post("/repositories", json={"path": "/no/such/place"})

    assert response.status_code == 422
    assert "Not a directory" in response.json()["detail"]


def test_register_rejects_an_unsafe_git_url(client):
    response = client.post("/repositories", json={"url": "file:///etc/passwd"})

    assert response.status_code == 422
    assert "https" in response.json()["detail"]


def test_reindex_endpoint(client, sample_repo):
    created = client.post("/repositories", json={"path": str(sample_repo)}).json()

    response = client.post(f"/repositories/{created['id']}/index")

    assert response.status_code == 202
    assert client.get(f"/repositories/{created['id']}/status").json()["index_status"] == "ready"


def test_reindex_requires_a_source_on_disk(client, indexed, db):
    from app.models import Repository

    repository = db.get(Repository, indexed.repository_id)
    repository.source_path = None
    db.commit()

    response = client.post(f"/repositories/{indexed.repository_id}/index")
    assert response.status_code == 422


def test_failed_indexing_is_visible_via_status(client, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    created = client.post("/repositories", json={"path": str(empty)}).json()
    body = client.get(f"/repositories/{created['id']}/status").json()

    assert body["index_status"] == "failed"
    assert "No indexable source files" in body["error_message"]


# ---------------------------------------------------------------------------
# ZIP upload
# ---------------------------------------------------------------------------

def test_upload_zip_indexes_the_archive(client):
    data = make_zip_bytes(
        {
            "proj/auth.py": "def login(user, password):\n    return check(user)\n",
            "proj/README.md": "# Project\n\nAuth notes.\n",
        }
    )

    response = client.post(
        "/repositories/upload",
        files={"archive": ("proj.zip", data, "application/zip")},
        data={"name": "uploaded"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["source_type"] == "zip"

    status_body = client.get(f"/repositories/{body['id']}/status").json()
    assert status_body["index_status"] == "ready"
    assert status_body["file_count"] == 2

    client.delete(f"/repositories/{body['id']}")


def test_upload_rejects_a_zip_slip_archive(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../../escaped.py", "owned = True\n")

    response = client.post(
        "/repositories/upload",
        files={"archive": ("evil.zip", buffer.getvalue(), "application/zip")},
    )

    assert response.status_code == 422
    assert "escapes the destination" in response.json()["detail"]


def test_upload_rejects_a_non_zip(client):
    response = client.post(
        "/repositories/upload",
        files={"archive": ("x.zip", b"definitely not a zip", "application/zip")},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Reading repositories
# ---------------------------------------------------------------------------

def test_list_and_get_repository(client, indexed):
    listed = client.get("/repositories").json()
    assert any(r["id"] == indexed.repository_id for r in listed)

    body = client.get(f"/repositories/{indexed.repository_id}").json()
    assert body["name"] == "demo"
    assert body["index_status"] == "ready"
    # The 1536-float embedding must never appear on the wire.
    assert "embedding" not in body


def test_missing_repository_is_404(client):
    assert client.get("/repositories/999999").status_code == 404
    assert client.get("/repositories/999999/status").status_code == 404
    assert client.get("/repositories/999999/files").status_code == 404


def test_list_files(client, indexed):
    files = client.get(f"/repositories/{indexed.repository_id}/files").json()

    paths = [f["path"] for f in files]
    assert "backend/auth.py" in paths
    assert not any("node_modules" in p for p in paths)


def test_list_files_filtered_by_language(client, indexed):
    files = client.get(
        f"/repositories/{indexed.repository_id}/files", params={"language": "python"}
    ).json()

    assert files
    assert {f["language"] for f in files} == {"python"}


def test_delete_repository(client, indexed):
    assert client.delete(f"/repositories/{indexed.repository_id}").status_code == 204
    assert client.get(f"/repositories/{indexed.repository_id}").status_code == 404


# ---------------------------------------------------------------------------
# Search endpoint (retrieval without generation)
# ---------------------------------------------------------------------------

def test_search_returns_ranked_hits(client, indexed):
    body = client.get(
        f"/repositories/{indexed.repository_id}/search",
        params={"q": "how are passwords hashed?", "top_k": 3},
    ).json()

    assert body["count"] == 3
    assert body["top_k"] == 3
    scores = [hit["score"] for hit in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert all(":" in hit["citation"] for hit in body["results"])


def test_search_accepts_filters(client, indexed):
    body = client.get(
        f"/repositories/{indexed.repository_id}/search",
        params={"q": "auth", "language": ["python"], "path_prefix": "backend/"},
    ).json()

    assert body["count"] > 0
    for hit in body["results"]:
        assert hit["language"] == "python"
        assert hit["file_path"].startswith("backend/")


def test_search_requires_a_query(client, indexed):
    response = client.get(f"/repositories/{indexed.repository_id}/search")
    assert response.status_code == 422


def test_search_on_an_unindexed_repository_is_409(client, sample_repo):
    created = client.post(
        "/repositories", json={"path": str(sample_repo), "index_now": False}
    ).json()

    response = client.get(
        f"/repositories/{created['id']}/search", params={"q": "anything"}
    )
    assert response.status_code == 409
    assert "not searchable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def test_chat_answers_with_citations(client, indexed, llm):
    body = client.post(
        "/chat",
        json={"repository_id": indexed.repository_id, "question": "where is auth?"},
    ).json()

    assert body["answer"] == ANSWER
    assert body["citations"][0]["index"] == 1
    assert ":" in body["citations"][0]["citation"]
    assert body["is_grounded"] is True
    assert body["conversation_id"] > 0
    assert llm.calls, "the model was never called"


def test_chat_creates_and_persists_a_conversation(client, db, indexed):
    body = client.post(
        "/chat",
        json={"repository_id": indexed.repository_id, "question": "where is auth?"},
    ).json()

    conversation = db.get(Conversation, body["conversation_id"])
    assert conversation.title == "where is auth?"
    assert [m.role for m in conversation.messages] == ["user", "assistant"]


def test_followup_replays_history_but_still_retrieves(client, indexed, llm):
    first = client.post(
        "/chat",
        json={"repository_id": indexed.repository_id, "question": "where is auth?"},
    ).json()

    second = client.post(
        "/chat",
        json={
            "repository_id": indexed.repository_id,
            "question": "how does it hash the password?",
            "conversation_id": first["conversation_id"],
        },
    ).json()

    assert second["conversation_id"] == first["conversation_id"]
    assert "where is auth?" in llm.last_prompt, "history was not replayed"
    assert second["sources"], "a follow-up answered without fresh retrieval"


def test_history_is_not_replayed_as_the_current_question(client, indexed, llm):
    """The new question must not also appear as a prior turn."""
    first = client.post(
        "/chat", json={"repository_id": indexed.repository_id, "question": "first?"}
    ).json()
    client.post(
        "/chat",
        json={
            "repository_id": indexed.repository_id,
            "question": "second?",
            "conversation_id": first["conversation_id"],
        },
    )

    roles = [m["role"] for m in llm.calls[-1]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_conversation_from_another_repository_is_rejected(client, db, sample_repo, indexed):
    other = index_repository(
        db,
        name="other",
        root=sample_repo,
        provider=FakeEmbeddingProvider(dimension=1536, model="fake-embedding"),
    )
    first = client.post(
        "/chat", json={"repository_id": indexed.repository_id, "question": "q"}
    ).json()

    response = client.post(
        "/chat",
        json={
            "repository_id": other.repository_id,
            "question": "q",
            "conversation_id": first["conversation_id"],
        },
    )

    assert response.status_code == 409, "answers would cite the wrong codebase"


def test_chat_on_missing_repository_is_404(client):
    response = client.post("/chat", json={"repository_id": 999999, "question": "q"})
    assert response.status_code == 404


def test_chat_rejects_an_empty_question(client, indexed):
    response = client.post(
        "/chat", json={"repository_id": indexed.repository_id, "question": ""}
    )
    assert response.status_code == 422


def test_chat_reports_no_context_without_calling_the_model(client, indexed, llm):
    body = client.post(
        "/chat",
        json={
            "repository_id": indexed.repository_id,
            "question": "where is the Stripe integration?",
            "languages": ["cobol"],
        },
    ).json()

    assert body["no_context"] is True
    assert body["citations"] == []
    assert llm.calls == []


def test_chat_strips_fabricated_citations(client, indexed, llm):
    llm._response = "Auth is here [1] and Stripe is in payments.py [9]."

    body = client.post(
        "/chat",
        json={"repository_id": indexed.repository_id, "question": "auth?"},
    ).json()

    assert body["invalid_markers"] == [9]
    assert "[9]" not in body["answer"]
    assert body["is_grounded"] is False


# ---------------------------------------------------------------------------
# Conversation endpoints
# ---------------------------------------------------------------------------

def test_list_and_fetch_conversations(client, indexed):
    created = client.post(
        "/chat", json={"repository_id": indexed.repository_id, "question": "where?"}
    ).json()

    listed = client.get(
        "/conversations", params={"repository_id": indexed.repository_id}
    ).json()
    assert [c["id"] for c in listed] == [created["conversation_id"]]

    detail = client.get(f"/conversations/{created['conversation_id']}").json()
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"


def test_delete_conversation_cascades_messages(client, db, indexed):
    created = client.post(
        "/chat", json={"repository_id": indexed.repository_id, "question": "q"}
    ).json()
    conversation_id = created["conversation_id"]

    assert client.delete(f"/conversations/{conversation_id}").status_code == 204
    assert client.get(f"/conversations/{conversation_id}").status_code == 404
    assert (
        db.query(Message).filter(Message.conversation_id == conversation_id).count() == 0
    )


def test_missing_conversation_is_404(client):
    assert client.get("/conversations/999999").status_code == 404
    assert client.delete("/conversations/999999").status_code == 404


def test_health_reports_the_model_actually_in_use(client):
    """With provider=fake, reporting 'text-embedding-3-small' would mislead."""
    body = client.get("/health").json()

    assert body["embedding_provider"] == "fake"
    assert body["embedding_model"] == "fake-embedding"
    assert body["llm_model"] == "fake-llm"


def test_health_survives_a_provider_that_cannot_be_built(client, monkeypatch):
    """A missing API key must still produce a diagnosis, not a 500."""
    import app.main as main_module

    def explode(settings):
        raise RuntimeError("OPENAI_API_KEY is empty")

    monkeypatch.setattr(main_module, "get_embedding_provider", explode)
    body = client.get("/health").json()

    assert body["embedding_model"].startswith("unavailable:")
    assert body["status"] == "ok"
