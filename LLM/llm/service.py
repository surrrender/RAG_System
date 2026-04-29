from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Iterator

from llm.config import Settings, load_settings
from llm.generator import OllamaGenerator
from llm.models import AnswerResult, ConversationTurn, RetrievalMetrics, RetrievedChunk
from llm.prompting import build_prompt
from llm.retrieval import Retriever


EMPTY_RESULT_ANSWER = "未找到足够依据，无法基于当前检索结果给出可靠答案。"


@dataclass(slots=True)
class PreparedAnswer:
    question: str
    chunks: list[RetrievedChunk]
    prompt: str
    retrieval_metrics: RetrievalMetrics
    prompt_build_ms: float | None = None


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

    def warm_up(self) -> None:
        self._retriever.warm_up()

    def answer_question(
        self, question: str, top_k: int, history: list[ConversationTurn] | None = None
    ) -> AnswerResult:
        prepared = self._prepare_answer(
            question=question,
            top_k=top_k,
            history=history,
        )

        if not prepared.chunks:
            return AnswerResult(
                question=prepared.question,
                answer=EMPTY_RESULT_ANSWER,
                citations=[],
                model=self._generator.model,
                retrieval_count=0,
            )

        answer = self._generator.generate(prepared.prompt)

        return AnswerResult(
            question=prepared.question,
            answer=answer,
            citations=prepared.chunks,
            model=self._generator.model,
            retrieval_count=len(prepared.chunks),
        )

    def stream_answer_question(
        self, question: str, top_k: int, history: list[ConversationTurn] | None = None
    ) -> Iterator[dict[str, object]]:
        start_time = time.perf_counter()
        prepared = self._prepare_answer(
            question=question,
            top_k=top_k,
            history=history,
        )
        retrieval_count = len(prepared.chunks)
        meta = {
            "question": prepared.question,
            "model": self._generator.model,
            "retrieval_count": retrieval_count,
            "server_started_at_ms": 0.0,
            "retrieval_finished_at_ms": _elapsed_ms(start_time),
        }
        _set_optional_metric(meta, "server_embed_ms", prepared.retrieval_metrics.embed_ms)
        _set_optional_metric(meta, "server_vector_search_ms", prepared.retrieval_metrics.vector_search_ms)
        _set_optional_metric(meta, "server_rerank_ms", prepared.retrieval_metrics.rerank_ms)
        _set_optional_metric(meta, "server_prompt_build_ms", prepared.prompt_build_ms)
        yield {
            "event": "meta",
            "data": meta,
        }

        if not prepared.chunks:
            yield {
                "event": "delta",
                "data": {
                    "text": EMPTY_RESULT_ANSWER,
                    "server_first_token_at_ms": _elapsed_ms(start_time),
                },
            }
            yield {"event": "citations", "data": {"citations": []}}
            yield {
                "event": "done",
                "data": {
                    "answer": EMPTY_RESULT_ANSWER,
                    "server_completed_at_ms": _elapsed_ms(start_time),
                },
            }
            return

        full_answer = ""
        first_token_reported = False
        for chunk in self._generator.generate_stream(prepared.prompt):
            full_answer += chunk
            data: dict[str, object] = {"text": chunk}
            if not first_token_reported:
                data["server_first_token_at_ms"] = _elapsed_ms(start_time)
                first_token_reported = True
            yield {"event": "delta", "data": data}

        yield {
            "event": "citations",
            "data": {"citations": [citation.to_dict() for citation in prepared.chunks]},
        }
        yield {
            "event": "done",
            "data": {
                "answer": full_answer,
                "server_completed_at_ms": _elapsed_ms(start_time),
            },
        }

    # 检索问题相关的知识库内容,构建 prompt,返回给生成器使用
    def _prepare_answer(
        self, question: str, top_k: int, history: list[ConversationTurn] | None = None
    ) -> PreparedAnswer:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")

        chunks, retrieval_metrics = self._retrieve(normalized_question, top_k)
        if not chunks:
            return PreparedAnswer(
                question=normalized_question,
                chunks=[],
                prompt="",
                retrieval_metrics=retrieval_metrics,
            )

        prompt_started_at = time.perf_counter()
        prompt = build_prompt(
            question=normalized_question,
            chunks=chunks,
            max_context_chars=self._max_context_chars,
            history=history,
        )
        return PreparedAnswer(
            question=normalized_question,
            chunks=chunks,
            prompt=prompt,
            retrieval_metrics=retrieval_metrics,
            prompt_build_ms=_elapsed_ms(prompt_started_at),
        )

    def _retrieve(self, question: str, top_k: int) -> tuple[list[RetrievedChunk], RetrievalMetrics]:
        retrieve_with_metrics = getattr(self._retriever, "retrieve_with_metrics", None)
        if callable(retrieve_with_metrics):
            return retrieve_with_metrics(question, top_k)
        return self._retriever.retrieve(question, top_k), RetrievalMetrics()


def build_service(settings: Settings | None = None) -> QAService:
    current = settings or load_settings()
    retriever = Retriever(
        qdrant_path=current.qdrant_path,
        qdrant_url=current.qdrant_url,
        qdrant_api_key=current.qdrant_api_key,
        collection_name=current.collection_name,
        embedder_provider=current.embedder_provider,
        embedding_model=current.embedding_model,
        embedding_device=current.embedding_device,
        reranker_provider=current.reranker_provider,
        reranker_model=current.reranker_model,
        reranker_device=current.reranker_device,
        rerank_candidate_limit=current.rerank_candidate_limit,
        disable_reranker=current.disable_reranker,
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


def _elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000, 3)


def _set_optional_metric(payload: dict[str, object], key: str, value: float | None) -> None:
    if value is not None:
        payload[key] = value
