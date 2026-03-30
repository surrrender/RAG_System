import json
from typing import Any

from llm.config import Settings, load_settings
from llm.models import ConversationTurn
from llm.service import QAService, build_service


def create_app(
    service: QAService | None = None,
    settings: Settings | None = None,
) -> Any:
    try:
        from fastapi import Body, FastAPI
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field, field_validator
    except ImportError as exc:
        raise RuntimeError(
            "fastapi and pydantic are required to run the HTTP API. Install the LLM project dependencies first."
        ) from exc

    current = settings or load_settings()
    qa_service = service or build_service(current)
    app = FastAPI(title="Local RAG QA API", version="0.1.0")

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
        question: str = Field(min_length=1)
        top_k: int = Field(default=current.top_k, ge=1, le=20)
        history: list[HistoryTurn] = Field(default_factory=list)

        @field_validator("question")
        @classmethod
        def validate_question(cls, value: str) -> str:
            stripped = value.strip()
            if not stripped:
                raise ValueError("question must not be empty")
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

    @app.post("/qa", response_model=QAResponse)
    def ask(payload: QARequest = Body(...)) -> QAResponse:
        history = [ConversationTurn(role=item.role, content=item.content) for item in payload.history]
        result = qa_service.answer_question(
            question=payload.question,
            top_k=payload.top_k,
            history=history,
        )
        return QAResponse.model_validate(result.to_dict())

    @app.post("/qa/stream")
    def ask_stream(payload: QARequest = Body(...)) -> StreamingResponse:

        history = [ConversationTurn(role=item.role, content=item.content) for item in payload.history]

        def event_stream() -> Any:
            try:
                for event in qa_service.stream_answer_question(
                    question=payload.question,
                    top_k=payload.top_k,
                    history=history,
                ):
                    yield _encode_sse(event=event["event"], data=event["data"])
            except Exception as exc:
                yield _encode_sse(event="error", data={"message": str(exc)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return app


def _encode_sse(event: object, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
