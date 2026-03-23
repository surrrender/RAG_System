from __future__ import annotations

from dataclasses import asdict, dataclass


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
class AnswerResult:
    question: str
    answer: str
    citations: list[RetrievedChunk]
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
