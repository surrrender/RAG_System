import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from llm.config import Settings, load_settings
from llm.models import Citation, ConversationSummary, ConversationTurn, StoredMessage
from llm.service import QAService, build_service
from llm.storage import ConversationNotFoundError, ConversationStore

logger = logging.getLogger(__name__)


def create_app(
    service: QAService | None = None,
    settings: Settings | None = None,
    store: ConversationStore | None = None,
) -> Any:
    try:
        from fastapi import Body, FastAPI, HTTPException, Query
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field, field_validator
    except ImportError as exc:
        raise RuntimeError(
            "fastapi and pydantic are required to run the HTTP API. Install the LLM project dependencies first."
        ) from exc

    current = settings or load_settings()
    qa_service = service or build_service(current)
    conversation_store = store or ConversationStore(current.sqlite_path)
    
    @asynccontextmanager
    async def lifespan(_: Any):
        warm_up = getattr(qa_service, "warm_up", None)
        if callable(warm_up):
            try:
                warm_up()
            except Exception:  # pragma: no cover - startup warmup failure is environment-dependent
                logger.exception("QA service warm-up failed during startup.")
        yield

    app = FastAPI(title="Local RAG QA API", version="0.1.0", lifespan=lifespan)

    class HistoryTurn(BaseModel):
        role: str
        content: str = Field(min_length=1)

        @field_validator("role")
        @classmethod
        def validate_role(cls, value: str) -> str:
            if value not in {"user", "assistant"}:
                raise ValueError("role must be user or assistant")
            return value

        @field_validator("content")
        @classmethod
        def validate_content(cls, value: str) -> str:
            stripped = value.strip()
            if not stripped:
                raise ValueError("content must not be empty")
            return stripped

    class QARequest(BaseModel):
        user_id: str = Field(min_length=1)
        conversation_id: str = Field(min_length=1)
        question: str = Field(min_length=1)
        top_k: int = Field(default=current.top_k, ge=1, le=20)
        history: list[HistoryTurn] = Field(default_factory=list)

        @field_validator("user_id", "conversation_id", "question")
        @classmethod
        def validate_required_text(cls, value: str) -> str:
            stripped = value.strip()
            if not stripped:
                raise ValueError("field must not be empty")
            return stripped

    class CreateConversationRequest(BaseModel):
        user_id: str = Field(min_length=1)

        @field_validator("user_id")
        @classmethod
        def validate_user_id(cls, value: str) -> str:
            stripped = value.strip()
            if not stripped:
                raise ValueError("user_id must not be empty")
            return stripped

    class RenameConversationRequest(CreateConversationRequest):
        title: str = Field(min_length=1)

        @field_validator("title")
        @classmethod
        def validate_title(cls, value: str) -> str:
            stripped = " ".join(value.strip().split())
            if not stripped:
                raise ValueError("title must not be empty")
            return stripped

    class CitationResponse(BaseModel):
        chunk_id: str
        score: float
        title: str | None = None
        url: str | None = None
        section_path: list[str] | None = None
        text: str | None = None

    class QAResponse(BaseModel):
        question: str
        answer: str
        citations: list[CitationResponse]
        model: str
        retrieval_count: int

    class ConversationResponse(BaseModel):
        id: str
        user_id: str
        title: str
        created_at: str
        updated_at: str
        last_message_at: str

    class MessageResponse(BaseModel):
        id: str
        conversation_id: str
        role: str
        content: str
        status: str
        citations: list[CitationResponse]
        model: str | None = None
        retrieval_count: int | None = None
        created_at: str

    @app.get("/conversations", response_model=list[ConversationResponse])
    def list_conversations(user_id: str = Query(..., min_length=1)) -> list[ConversationResponse]:
        return [
            ConversationResponse.model_validate(item.to_dict())
            for item in conversation_store.list_conversations(user_id=user_id.strip())
        ]

    @app.post("/conversations", response_model=ConversationResponse)
    def create_conversation(payload: CreateConversationRequest = Body(...)) -> ConversationResponse:
        conversation = conversation_store.create_conversation(user_id=payload.user_id)
        return ConversationResponse.model_validate(conversation.to_dict())

    @app.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
    def rename_conversation(
        conversation_id: str,
        payload: RenameConversationRequest = Body(...),
    ) -> ConversationResponse:
        try:
            conversation = conversation_store.rename_conversation(
                user_id=payload.user_id,
                conversation_id=conversation_id,
                title=payload.title,
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ConversationResponse.model_validate(conversation.to_dict())

    @app.delete("/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str, user_id: str = Query(..., min_length=1)) -> None:
        try:
            conversation_store.delete_conversation(user_id=user_id.strip(), conversation_id=conversation_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
    def get_messages(conversation_id: str, user_id: str = Query(..., min_length=1)) -> list[MessageResponse]:
        try:
            messages = conversation_store.get_messages(user_id=user_id.strip(), conversation_id=conversation_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [MessageResponse.model_validate(item.to_dict()) for item in messages]

    @app.post("/qa", response_model=QAResponse)
    def ask(payload: QARequest = Body(...)) -> QAResponse:
        try:
            history = _load_conversation_history(
                store=conversation_store,
                user_id=payload.user_id,
                conversation_id=payload.conversation_id,
                fallback_history=payload.history,
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        conversation_store.add_message(
            user_id=payload.user_id,
            conversation_id=payload.conversation_id,
            role="user",
            content=payload.question,
            status="done",
        )
        result = qa_service.answer_question(
            question=payload.question,
            top_k=payload.top_k,
            history=history,
        )
        conversation_store.add_message(
            user_id=payload.user_id,
            conversation_id=payload.conversation_id,
            role="assistant",
            content=result.answer,
            status="done",
            citations=[Citation(**item.to_dict()) for item in result.citations],
            model=result.model,
            retrieval_count=result.retrieval_count,
        )
        return QAResponse.model_validate(result.to_dict())

    @app.post("/qa/stream")
    def ask_stream(payload: QARequest = Body(...)) -> StreamingResponse:
        try:
            history = _load_conversation_history(
                store=conversation_store,
                user_id=payload.user_id,
                conversation_id=payload.conversation_id,
                fallback_history=payload.history,
            )
            conversation_store.ensure_conversation(user_id=payload.user_id, conversation_id=payload.conversation_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        conversation_store.add_message(
            user_id=payload.user_id,
            conversation_id=payload.conversation_id,
            role="user",
            content=payload.question,
            status="done",
        )
        assistant_message = conversation_store.add_message(
            user_id=payload.user_id,
            conversation_id=payload.conversation_id,
            role="assistant",
            content="",
            status="streaming",
        )

        def event_stream() -> Any:
            citations: list[Citation] = []
            model: str | None = None
            retrieval_count: int | None = None
            answer_chunks: list[str] = []
            try:
                for event in qa_service.stream_answer_question(
                    question=payload.question,
                    top_k=payload.top_k,
                    history=history,
                ):
                    if event["event"] == "meta":
                        data = dict(event["data"])
                        model = str(data.get("model") or "") or None
                        retrieval_count_value = data.get("retrieval_count")
                        retrieval_count = int(retrieval_count_value) if retrieval_count_value is not None else None
                    elif event["event"] == "delta":
                        answer_text = str(event["data"].get("text") or "")
                        answer_chunks.append(answer_text)
                    elif event["event"] == "citations":
                        citations = [Citation(**item) for item in event["data"].get("citations", [])]
                    elif event["event"] == "error":
                        message = str(event["data"].get("message") or "生成失败")
                        conversation_store.update_message(
                            user_id=payload.user_id,
                            conversation_id=payload.conversation_id,
                            message_id=assistant_message.id,
                            content=message,
                            status="error",
                            citations=[],
                            model=model,
                            retrieval_count=retrieval_count,
                        )
                    elif event["event"] == "done":
                        final_answer = str(event["data"].get("answer") or "".join(answer_chunks))
                        conversation_store.update_message(
                            user_id=payload.user_id,
                            conversation_id=payload.conversation_id,
                            message_id=assistant_message.id,
                            content=final_answer,
                            status="done",
                            citations=citations,
                            model=model,
                            retrieval_count=retrieval_count,
                        )
                    yield _encode_sse(event=event["event"], data=event["data"])
            except Exception as exc:
                conversation_store.update_message(
                    user_id=payload.user_id,
                    conversation_id=payload.conversation_id,
                    message_id=assistant_message.id,
                    content=str(exc),
                    status="error",
                    citations=[],
                    model=model,
                    retrieval_count=retrieval_count,
                )
                yield _encode_sse(event="error", data={"message": str(exc)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return app


def _encode_sse(event: object, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _load_conversation_history(
    store: ConversationStore,
    user_id: str,
    conversation_id: str,
    fallback_history: list[object],
) -> list[ConversationTurn]:
    stored_messages = store.get_messages(user_id=user_id, conversation_id=conversation_id)
    if stored_messages:
        return [
            ConversationTurn(role=item.role, content=item.content)
            for item in stored_messages
            if item.role in {"user", "assistant"} and item.content.strip() and item.status != "error"
        ]
    return [ConversationTurn(role=item.role, content=item.content) for item in fallback_history]
