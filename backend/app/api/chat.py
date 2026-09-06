"""Chat endpoints: ask questions, keep conversations.

Conversation history is persisted so follow-ups work across requests, but it is
never a shortcut past retrieval (spec section 14). Every question runs its own
search; history only helps the model resolve what "it" refers to.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_config, get_db, get_embeddings, get_llm
from app.config import Settings
from app.models import Conversation, Message, Repository
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetailOut,
    ConversationOut,
)
from app.services.embeddings import EmbeddingProvider
from app.services.generator import Generator
from app.services.llm import LLMError, LLMProvider
from app.services.retriever import RetrievalError, SearchFilters

router = APIRouter(tags=["chat"])

# A conversation title is just the opening question, trimmed. Enough to tell
# threads apart in a sidebar without spending an LLM call to summarise.
TITLE_LENGTH = 80


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    session: Session = Depends(get_db),
    embeddings: EmbeddingProvider = Depends(get_embeddings),
    llm: LLMProvider = Depends(get_llm),
    settings: Settings = Depends(get_config),
) -> ChatResponse:
    repository = session.get(Repository, payload.repository_id)
    if repository is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Repository {payload.repository_id} not found",
        )

    conversation = _resolve_conversation(session, payload, repository)

    # Load history BEFORE storing the new question, or the question would be
    # replayed to the model as if it were a previous turn.
    history = [
        {"role": message.role, "content": message.content}
        for message in conversation.messages
    ]

    filters = SearchFilters(
        languages=payload.languages, path_prefix=payload.path_prefix
    )
    generator = Generator(
        session, llm=llm, embedding_provider=embeddings, settings=settings
    )

    try:
        result = generator.answer(
            payload.question,
            payload.repository_id,
            history=history,
            top_k=payload.top_k,
            filters=None if filters.is_empty() else filters,
        )
    except RetrievalError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    session.add_all(
        [
            Message(conversation_id=conversation.id, role="user", content=payload.question),
            Message(
                conversation_id=conversation.id, role="assistant", content=result.answer
            ),
        ]
    )
    session.commit()

    payload_dict = result.as_dict()
    return ChatResponse(
        conversation_id=conversation.id,
        repository_id=payload.repository_id,
        question=result.question,
        answer=result.answer,
        citations=payload_dict["citations"],
        sources=payload_dict["sources"],
        is_grounded=result.is_grounded,
        no_context=result.no_context,
        citation_coverage=result.citation_coverage,
        invalid_markers=result.invalid_markers,
        model=result.model,
        usage=payload_dict["usage"],
    )


def _resolve_conversation(
    session: Session, payload: ChatRequest, repository: Repository
) -> Conversation:
    if payload.conversation_id is None:
        conversation = Conversation(
            repository_id=repository.id,
            title=payload.question[:TITLE_LENGTH],
        )
        session.add(conversation)
        session.commit()
        return conversation

    conversation = session.get(Conversation, payload.conversation_id)
    if conversation is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Conversation {payload.conversation_id} not found",
        )
    if conversation.repository_id != repository.id:
        # Replaying one repository's history against another's code would
        # produce answers that cite the wrong codebase entirely.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Conversation belongs to a different repository",
        )
    return conversation


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(
    repository_id: int | None = Query(default=None),
    session: Session = Depends(get_db),
) -> list[Conversation]:
    statement = select(Conversation).order_by(Conversation.created_at.desc())
    if repository_id is not None:
        statement = statement.where(Conversation.repository_id == repository_id)
    return list(session.scalars(statement).all())


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
def get_conversation(
    conversation_id: int, session: Session = Depends(get_db)
) -> Conversation:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Conversation {conversation_id} not found"
        )
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int, session: Session = Depends(get_db)
) -> None:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Conversation {conversation_id} not found"
        )
    session.delete(conversation)
    session.commit()
