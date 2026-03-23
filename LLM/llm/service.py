from __future__ import annotations

from llm.config import Settings, load_settings
from llm.generator import OllamaGenerator
from llm.models import AnswerResult
from llm.prompting import build_prompt
from llm.retrieval import Retriever


EMPTY_RESULT_ANSWER = "未找到足够依据，无法基于当前检索结果给出可靠答案。"


class QAService:
    def __init__(
        self,
        retriever: Retriever,
        generator: OllamaGenerator,
        max_context_chars: int,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._max_context_chars = max_context_chars

    def answer_question(self, question: str, top_k: int) -> AnswerResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")

        chunks = self._retriever.retrieve(normalized_question, top_k=top_k)
        if not chunks:
            return AnswerResult(
                question=normalized_question,
                answer=EMPTY_RESULT_ANSWER,
                citations=[],
                model=self._generator.model,
                retrieval_count=0,
            )

        prompt = build_prompt(
            question=normalized_question,
            chunks=chunks,
            max_context_chars=self._max_context_chars,
        )
        answer = self._generator.generate(prompt)
        return AnswerResult(
            question=normalized_question,
            answer=answer,
            citations=chunks,
            model=self._generator.model,
            retrieval_count=len(chunks),
        )


def build_service(settings: Settings | None = None) -> QAService:
    current = settings or load_settings()
    retriever = Retriever(
        qdrant_path=current.qdrant_path,
        collection_name=current.collection_name,
        embedder_provider=current.embedder_provider,
        embedding_model=current.embedding_model,
    )
    generator = OllamaGenerator(
        host=current.ollama_host,
        model=current.generation_model,
        timeout=current.request_timeout,
    )
    return QAService(
        retriever=retriever,
        generator=generator,
        max_context_chars=current.max_context_chars,
    )


def answer_question(question: str, top_k: int | None = None, settings: Settings | None = None) -> AnswerResult:
    current = settings or load_settings()
    service = build_service(current)
    return service.answer_question(question=question, top_k=top_k or current.top_k)
