"""Repository endpoints: register, index, inspect, search, delete.

HOW LONG-RUNNING INDEXING IS HANDLED (spec section 30)
------------------------------------------------------
Indexing a real repository takes anywhere from seconds to minutes: it parses
every file, chunks it, and makes network calls to embed thousands of chunks.
Doing that inside the request would time out any normal client.

So the write is split. POST returns a repository id and `index_status:
"pending"` immediately, the work runs in a background task, and the client
polls GET /repositories/{id}/status until the status becomes `ready` or
`failed`. Because the status lives in the database rather than in process
memory, a client can also poll it after a reconnect, and a crash leaves a row
that says `failed` with the reason rather than one stuck on `indexing`.

FastAPI's BackgroundTasks is the right tool at this size: one process, work
measured in minutes. A production system with many concurrent indexes would
move this to a real queue (Celery, RQ, arq) to get retries, concurrency limits
and durability across restarts. That is a deliberate deferral, not an
oversight.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_background_session, get_config, get_db, get_embeddings
from app.config import Settings
from app.models import File as FileModel
from app.models import Repository
from app.schemas import (
    FileOut,
    IndexStatusOut,
    RepositoryCreate,
    RepositoryOut,
    SearchResponse,
)
from app.services.embeddings import EmbeddingProvider
from app.services.indexer import create_repository, delete_repository, run_indexing
from app.services.retriever import RetrievalError, Retriever, SearchFilters
from app.services.sources import (
    SourceError,
    clone_repository,
    cleanup_storage,
    extract_zip,
    repository_name_from_url,
    storage_path_for,
)

router = APIRouter(prefix="/repositories", tags=["repositories"])


def _load(session: Session, repository_id: int) -> Repository:
    repository = session.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository {repository_id} not found",
        )
    return repository


def _index_in_background(
    session_factory: Callable[[], AbstractContextManager[Session]],
    repository_id: int,
    provider: EmbeddingProvider,
    settings: Settings,
) -> None:
    """Background entry point. Must never raise into the event loop.

    run_indexing already records failures on the repository row, so the client
    polling /status learns what went wrong. Re-raising here would only produce
    an unhandled task exception nobody reads.
    """
    with session_factory() as session:
        repository = session.get(Repository, repository_id)
        if repository is None:
            return
        try:
            run_indexing(session, repository, provider=provider, settings=settings)
        except Exception:
            pass


@router.post("", response_model=RepositoryOut, status_code=status.HTTP_201_CREATED)
def register_repository(
    payload: RepositoryCreate,
    background: BackgroundTasks,
    session: Session = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embeddings),
    settings: Settings = Depends(get_config),
    session_factory=Depends(get_background_session),
) -> Repository:
    """Register a repository from a git URL or a local path."""
    if bool(payload.url) == bool(payload.path):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide exactly one of 'url' or 'path'",
        )

    if payload.url:
        try:
            name = payload.name or repository_name_from_url(payload.url)
        except SourceError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        source_type, source_path = "github", None
    else:
        root = Path(payload.path).expanduser()
        if not root.is_dir():
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Not a directory: {payload.path}",
            )
        name = payload.name or root.resolve().name
        source_type, source_path = "local", str(root.resolve())

    repository = create_repository(
        session,
        name=name,
        url=payload.url,
        source_type=source_type,
        source_path=source_path,
        provider=provider,
        chunk_strategy=payload.chunk_strategy,
        max_chunk_lines=payload.max_chunk_lines,
        settings=settings,
    )

    if payload.url:
        # Cloning is slow and can fail, so it belongs in the background with
        # the indexing rather than blocking the response.
        background.add_task(
            _clone_then_index,
            session_factory,
            repository.id,
            payload.url,
            provider,
            settings,
        )
    elif payload.index_now:
        background.add_task(
            _index_in_background, session_factory, repository.id, provider, settings
        )

    return repository


def _clone_then_index(
    session_factory: Callable[[], AbstractContextManager[Session]],
    repository_id: int,
    url: str,
    provider: EmbeddingProvider,
    settings: Settings,
) -> None:
    with session_factory() as session:
        repository = session.get(Repository, repository_id)
        if repository is None:
            return
        try:
            repository.index_status = "indexing"
            session.commit()
            path = clone_repository(url, storage_path_for(repository_id))
            repository.source_path = str(path)
            session.commit()
        except SourceError as exc:
            session.rollback()
            repository = session.get(Repository, repository_id)
            if repository is not None:
                repository.index_status = "failed"
                repository.error_message = str(exc)[:2000]
                session.commit()
            return

        try:
            run_indexing(session, repository, provider=provider, settings=settings)
        except Exception:
            pass


@router.post("/upload", response_model=RepositoryOut, status_code=status.HTTP_201_CREATED)
async def upload_repository(
    background: BackgroundTasks,
    archive: UploadFile = File(..., description="ZIP archive of the codebase"),
    name: str | None = Form(default=None),
    chunk_strategy: str = Form(default="code_aware"),
    session: Session = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embeddings),
    settings: Settings = Depends(get_config),
    session_factory=Depends(get_background_session),
) -> Repository:
    """Register a repository from an uploaded ZIP archive."""
    if chunk_strategy not in ("code_aware", "naive"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Unknown chunk_strategy")

    repository = create_repository(
        session,
        name=name or Path(archive.filename or "upload").stem,
        source_type="zip",
        provider=provider,
        chunk_strategy=chunk_strategy,
        settings=settings,
    )

    # Spool to a temp file rather than holding the whole archive in memory.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as spool:
        temp_path = Path(spool.name)
        while data := await archive.read(1024 * 1024):
            spool.write(data)

    try:
        root = extract_zip(temp_path, storage_path_for(repository.id))
    except SourceError as exc:
        repository.index_status = "failed"
        repository.error_message = str(exc)[:2000]
        session.commit()
        cleanup_storage(repository.id)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    repository.source_path = str(root)
    session.commit()

    background.add_task(
        _index_in_background, session_factory, repository.id, provider, settings
    )
    return repository


@router.get("", response_model=list[RepositoryOut])
def list_repositories_endpoint(session: Session = Depends(get_db)) -> list[Repository]:
    return list(
        session.scalars(select(Repository).order_by(Repository.created_at.desc())).all()
    )


@router.get("/{repository_id}", response_model=RepositoryOut)
def get_repository(repository_id: int, session: Session = Depends(get_db)) -> Repository:
    return _load(session, repository_id)


@router.get("/{repository_id}/status", response_model=IndexStatusOut)
def get_index_status(repository_id: int, session: Session = Depends(get_db)) -> IndexStatusOut:
    """Poll this while indexing runs."""
    repository = _load(session, repository_id)
    return IndexStatusOut(
        repository_id=repository.id,
        index_status=repository.index_status,
        file_count=repository.file_count,
        chunk_count=repository.chunk_count,
        error_message=repository.error_message,
    )


@router.post("/{repository_id}/index", response_model=IndexStatusOut, status_code=status.HTTP_202_ACCEPTED)
def index_repository_endpoint(
    repository_id: int,
    background: BackgroundTasks,
    session: Session = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embeddings),
    settings: Settings = Depends(get_config),
    session_factory=Depends(get_background_session),
) -> IndexStatusOut:
    """(Re)index an already-registered repository."""
    repository = _load(session, repository_id)

    if repository.index_status == "indexing":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Repository {repository_id} is already indexing",
        )
    if not repository.source_path:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Repository {repository_id} has no source on disk to index",
        )

    repository.index_status = "pending"
    repository.error_message = None
    session.commit()

    background.add_task(
        _index_in_background, session_factory, repository.id, provider, settings
    )
    return IndexStatusOut(
        repository_id=repository.id,
        index_status="pending",
        file_count=repository.file_count,
        chunk_count=repository.chunk_count,
    )


@router.get("/{repository_id}/files", response_model=list[FileOut])
def list_files(
    repository_id: int,
    language: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> list[FileModel]:
    _load(session, repository_id)

    statement = select(FileModel).where(FileModel.repository_id == repository_id)
    if language:
        statement = statement.where(FileModel.language == language)

    return list(
        session.scalars(
            statement.order_by(FileModel.path).limit(limit).offset(offset)
        ).all()
    )


@router.get("/{repository_id}/search", response_model=SearchResponse)
def search(
    repository_id: int,
    q: str = Query(..., min_length=1, description="Natural language query"),
    top_k: int | None = Query(default=None, ge=1, le=100),
    language: list[str] | None = Query(default=None),
    kind: list[str] | None = Query(default=None),
    path_prefix: str | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=-1.0, le=1.0),
    session: Session = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embeddings),
    settings: Settings = Depends(get_config),
) -> SearchResponse:
    """Retrieval WITHOUT generation (spec section 16).

    The most useful debugging endpoint in the system. When an answer is wrong
    there are only two candidates: retrieval fetched the wrong code, or the LLM
    misread the right code. This separates them, and they need entirely
    different fixes.
    """
    _load(session, repository_id)

    filters = SearchFilters(
        languages=language,
        kinds=kind,
        path_prefix=path_prefix,
        min_score=min_score,
    )
    retriever = Retriever(session, provider=provider, settings=settings)

    try:
        result = retriever.search(
            q, repository_id, top_k=top_k, filters=None if filters.is_empty() else filters
        )
    except RetrievalError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    return SearchResponse(**result.as_dict())


@router.delete("/{repository_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_repository_endpoint(
    repository_id: int, session: Session = Depends(get_db)
) -> None:
    _load(session, repository_id)
    delete_repository(session, repository_id)
    cleanup_storage(repository_id)
