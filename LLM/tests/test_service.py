from llm.generator import OllamaGenerator
from llm.models import RetrievedChunk
from llm.service import EMPTY_RESULT_ANSWER, QAService


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((question, top_k))
        return self._chunks


class StubGenerator:
    def __init__(self, text: str) -> None:
        self.model = "stub-model"
        self._text = text
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._text


def test_service_returns_conservative_answer_when_no_chunks() -> None:
    service = QAService(retriever=StubRetriever([]), generator=StubGenerator("ignored"), max_context_chars=1000)

    result = service.answer_question("什么是 App？", top_k=5)

    assert result.answer == EMPTY_RESULT_ANSWER
    assert result.citations == []
    assert result.retrieval_count == 0


def test_service_returns_answer_and_citations() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            score=0.8,
            title="App",
            url="https://example.com/app",
            section_path=["生命周期"],
            text="App 包含 onLaunch。",
        )
    ]
    retriever = StubRetriever(chunks)
    generator = StubGenerator("App 生命周期包括 onLaunch。")
    service = QAService(retriever=retriever, generator=generator, max_context_chars=1000)

    result = service.answer_question("App 生命周期是什么？", top_k=3)

    assert retriever.calls == [("App 生命周期是什么？", 3)]
    assert generator.prompts
    assert result.answer == "App 生命周期包括 onLaunch。"
    assert result.citations == chunks
    assert result.retrieval_count == 1
