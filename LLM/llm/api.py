from typing import Any

from llm.config import Settings, load_settings
from llm.service import QAService, build_service


def create_app(
    service: QAService | None = None,
    settings: Settings | None = None,
) -> Any:
    try:
        from fastapi import Body, FastAPI
        from pydantic import BaseModel, Field, field_validator
    except ImportError as exc:
        raise RuntimeError(
            "fastapi and pydantic are required to run the HTTP API. Install the LLM project dependencies first."
        ) from exc

    current = settings or load_settings()
    qa_service = service or build_service(current)
    app = FastAPI(title="Local RAG QA API", version="0.1.0")

    class QARequest(BaseModel):
        question: str = Field(min_length=1)
        top_k: int = Field(default=current.top_k, ge=1, le=20)

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
        result = qa_service.answer_question(question=payload.question, top_k=payload.top_k)
        return QAResponse.model_validate(result.to_dict())

    return app
