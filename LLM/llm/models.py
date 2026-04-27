from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    title: str | None
    url: str | None
    section_path: list[str] | None
    text: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class Citation:
    chunk_id: str
    score: float
    title: str | None
    url: str | None
    section_path: list[str] | None
    text: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str
    citations: list[Citation]
    model: str
    retrieval_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [item.to_dict() for item in self.citations],
            "model": self.model,
            "retrieval_count": self.retrieval_count,
        }


@dataclass(slots=True)
class ConversationSummary:
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    last_message_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class StoredMessage:
    id: str
    conversation_id: str
    role: str
    content: str
    status: str
    model: str | None
    retrieval_count: int | None
    citations: list[Citation]
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "status": self.status,
            "model": self.model,
            "retrieval_count": self.retrieval_count,
            "citations": [item.to_dict() for item in self.citations],
            "created_at": self.created_at,
        }
