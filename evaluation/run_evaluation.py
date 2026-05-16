from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer
from typing_extensions import Annotated

app = typer.Typer(
    name="evaluation",
    help="RAG 系统评估工具 — 检索评估 (Recall@K, MRR, Hit Rate) + 问答评估 (BLEU, ROUGE, Faithfulness, Human Score)",
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CRAWLER_OUTPUTS = PROJECT_ROOT / "Crawler" / "outputs"
DEFAULT_CHUNKS_JSONL = CRAWLER_OUTPUTS / "framework_chunks.jsonl"
DEFAULT_QUESTIONS_FILE = Path(__file__).resolve().parent / "data" / "test_questions.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_OUTPUT_HTML = DEFAULT_OUTPUT_DIR / "evaluation_report.html"


@app.command()
def generate(
    chunks_jsonl: Annotated[
        Path,
        typer.Option("--chunks-jsonl", "-c", help="JSONL 文件路径，包含所有 chunk 数据"),
    ] = DEFAULT_CHUNKS_JSONL,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="输出的测试问题 JSON 文件路径"),
    ] = DEFAULT_QUESTIONS_FILE,
    num_questions: Annotated[
        int,
        typer.Option("--num-questions", "-n", help="生成的 QA 对数量"),
    ] = 50,
    seed: Annotated[int, typer.Option("--seed", "-s", help="随机种子")] = 42,
):
    """生成测试 QA 数据集（通过 LLM 从 chunk 中自动生成）"""
    from evaluation.generators.question_generator import generate_questions
    from llm.config import load_settings
    from llm.generator import OllamaGenerator
    from llm.networking import is_local_service_url, normalize_local_service_url

    settings = load_settings()
    print(f"使用 LLM: {settings.generation_model} @ {settings.ollama_host}")
    print(f"从 chunk 文件采样: {chunks_jsonl}")

    generator_client = OllamaGenerator(
        host=settings.ollama_host,
        model=settings.generation_model,
        timeout=settings.request_timeout,
    )

    questions = generate_questions(
        chunks_jsonl=chunks_jsonl,
        generator_client=generator_client,
        num_questions=num_questions,
        output_path=output,
        random_seed=seed,
    )

    print(f"\n已生成 {len(questions)} 条 QA 对，保存至: {output}")


@app.command()
def retrieval(
    questions_file: Annotated[
        Path,
        typer.Option("--questions-file", "-q", help="测试问题 JSON 文件"),
    ] = DEFAULT_QUESTIONS_FILE,
    results_file: Annotated[
        Path | None,
        typer.Option("--results-file", "-r", help="保存检索结果 JSON (可选)"),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="检索评估用的 top_k")] = 10,
):
    """只运行检索评估"""
    from evaluation.evaluators.retrieval_evaluator import evaluate_retrieval
    from evaluation.generators.question_generator import load_questions
    from llm.config import load_settings

    settings = load_settings()
    questions = load_questions(questions_file)
    print(f"加载 {len(questions)} 条测试问题")
    print(f"检索评估 top_k={top_k}")

    from llm.retrieval import Retriever

    retriever = Retriever(
        qdrant_path=settings.qdrant_path,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
        embedder_provider=settings.embedder_provider,
        embedding_model=settings.embedding_model,
        embedding_device=settings.embedding_device,
        reranker_provider=settings.reranker_provider,
        reranker_model=settings.reranker_model,
        reranker_device=settings.reranker_device,
        rerank_candidate_limit=settings.rerank_candidate_limit,
        disable_reranker=settings.disable_reranker,
    )

    result = evaluate_retrieval(retriever, questions, eval_top_k=top_k)

    _print_retrieval_summary(result)

    if results_file:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_file.write_text(
            json.dumps(
                {
                    "avg_recall_at_5": result.avg_recall_at_5,
                    "avg_recall_at_10": result.avg_recall_at_10,
                    "avg_mrr": result.avg_mrr,
                    "hit_rate_at_5": result.hit_rate_at_5,
                    "hit_rate_at_10": result.hit_rate_at_10,
                    "per_question": [
                        {
                            "id": q.question_id,
                            "recall_at_1": q.recall_at_1,
                            "recall_at_5": q.recall_at_5,
                            "mrr": q.mrr,
                            "hit_at_5": q.hit_at_5,
                        }
                        for q in result.questions
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"检索结果已保存: {results_file}")


@app.command()
def qa(
    questions_file: Annotated[
        Path,
        typer.Option("--questions-file", "-q", help="测试问题 JSON 文件"),
    ] = DEFAULT_QUESTIONS_FILE,
    results_file: Annotated[
        Path | None,
        typer.Option("--results-file", "-r", help="保存问答结果 JSON (可选)"),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="RAG 检索 top_k")] = 5,
):
    """只运行问答评估"""
    from evaluation.evaluators.qa_evaluator import evaluate_qa
    from evaluation.generators.question_generator import load_questions
    from llm.config import load_settings
    from llm.generator import OllamaGenerator
    from llm.service import build_service

    settings = load_settings()
    questions = load_questions(questions_file)
    print(f"加载 {len(questions)} 条测试问题")
    print(f"使用 LLM: {settings.generation_model} @ {settings.ollama_host}")
    print(f"RAG top_k={top_k}")

    service = build_service(settings)
    generator_client = OllamaGenerator(
        host=settings.ollama_host,
        model=settings.generation_model,
        timeout=settings.request_timeout,
    )

    result = evaluate_qa(service, generator_client, questions)

    _print_qa_summary(result)

    if results_file:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_file.write_text(
            json.dumps(
                {
                    "avg_bleu_4": result.avg_bleu_4,
                    "avg_rouge_l_f": result.avg_rouge_l_f,
                    "avg_faithfulness": result.avg_faithfulness,
                    "avg_overall": result.avg_overall,
                    "per_question": [
                        {
                            "id": q.question_id,
                            "bleu_4": q.bleu_4,
                            "rouge_l_f": q.rouge_l_f,
                            "faithfulness": q.faithfulness_score,
                            "overall": q.overall_score,
                        }
                        for q in result.questions
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"问答结果已保存: {results_file}")


@app.command()
def full(
    questions_file: Annotated[
        Path,
        typer.Option("--questions-file", "-q", help="测试问题 JSON 文件"),
    ] = DEFAULT_QUESTIONS_FILE,
    output_html: Annotated[
        Path,
        typer.Option("--output-html", "-o", help="HTML 报告输出路径"),
    ] = DEFAULT_OUTPUT_HTML,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="RAG 检索 top_k")] = 5,
    retrieval_top_k: Annotated[int, typer.Option("--retrieval-top-k", help="检索评估 top_k")] = 10,
):
    """运行完整评估并生成 HTML 报告"""
    _check_prerequisites()

    print("=" * 60)
    print("  RAG 系统评估 — 完整运行")
    print("=" * 60)

    # Step 0: Load settings
    from llm.config import load_settings

    settings = load_settings()
    model = settings.generation_model
    print(f"\n配置: model={model}, ollama={settings.ollama_host}")

    # Step 1: Load questions
    from evaluation.generators.question_generator import load_questions

    if not questions_file.exists():
        print(f"\n问题文件 {questions_file} 不存在。请先运行 'generate' 命令生成 QA 数据集。")
        raise typer.Exit(1)

    questions = load_questions(questions_file)
    print(f"\n加载 {len(questions)} 条测试问题")

    # Build dataset stats
    from embedding_indexing.io import iter_chunks

    all_chunks = list(iter_chunks(DEFAULT_CHUNKS_JSONL))
    dataset_stats = {
        "total_documents": len(set(c.doc_id for c in all_chunks)),
        "total_chunks": len(all_chunks),
        "total_questions": len(questions),
        "generation_model": model,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "questions": [
            {
                "question": q.question,
                "category": q.category,
                "difficulty": q.difficulty,
                "relevant_chunk_ids": q.relevant_chunk_ids,
            }
            for q in questions
        ],
    }

    # Step 2: Retrieval evaluation
    print(f"\n{'—' * 40}")
    print("阶段 1/2: 检索评估 (Recall@K, MRR, Hit Rate)")
    print(f"{'—' * 40}")

    from evaluation.evaluators.retrieval_evaluator import evaluate_retrieval
    from llm.retrieval import Retriever

    retriever = Retriever(
        qdrant_path=settings.qdrant_path,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
        embedder_provider=settings.embedder_provider,
        embedding_model=settings.embedding_model,
        embedding_device=settings.embedding_device,
        reranker_provider=settings.reranker_provider,
        reranker_model=settings.reranker_model,
        reranker_device=settings.reranker_device,
        rerank_candidate_limit=settings.rerank_candidate_limit,
        disable_reranker=settings.disable_reranker,
    )

    retrieval_result = evaluate_retrieval(retriever, questions, eval_top_k=retrieval_top_k)
    _print_retrieval_summary(retrieval_result)

    # Step 3: QA evaluation
    print(f"\n{'—' * 40}")
    print("阶段 2/2: 问答评估 (BLEU, ROUGE, Faithfulness, Human Score)")
    print(f"{'—' * 40}")

    from evaluation.evaluators.qa_evaluator import evaluate_qa
    from llm.generator import OllamaGenerator
    from llm.service import build_service

    service = build_service(settings)
    generator_client = OllamaGenerator(
        host=settings.ollama_host,
        model=settings.generation_model,
        timeout=settings.request_timeout,
    )

    qa_result = evaluate_qa(service, generator_client, questions)
    _print_qa_summary(qa_result)

    # Step 4: Generate HTML report
    print(f"\n{'—' * 40}")
    print("生成 HTML 报告")
    print(f"{'—' * 40}")

    from evaluation.reporters.html_reporter import generate_html_report

    generate_html_report(
        output_path=output_html,
        dataset_stats=dataset_stats,
        retrieval_result=retrieval_result,
        qa_result=qa_result,
    )

    print(f"\n报告已生成: {output_html.resolve()}")
    print(f"\n在浏览器中打开: open {output_html.resolve()}")


def _check_prerequisites() -> None:
    issues: list[str] = []

    if not DEFAULT_CHUNKS_JSONL.exists():
        issues.append(f"Chunk 文件不存在: {DEFAULT_CHUNKS_JSONL}")

    try:
        import ollama
    except ImportError:
        issues.append("ollama 未安装，无法调用 LLM")

    if issues:
        for issue in issues:
            print(f"错误: {issue}", file=sys.stderr)
        raise typer.Exit(1)


def _print_retrieval_summary(result) -> None:
    print(f"\n检索评估结果:")
    print(f"  Recall@1:    {result.avg_recall_at_1:.4f}")
    print(f"  Recall@3:    {result.avg_recall_at_3:.4f}")
    print(f"  Recall@5:    {result.avg_recall_at_5:.4f}")
    print(f"  Recall@10:   {result.avg_recall_at_10:.4f}")
    print(f"  MRR:         {result.avg_mrr:.4f}")
    print(f"  Hit Rate@1:  {result.hit_rate_at_1:.4f}")
    print(f"  Hit Rate@5:  {result.hit_rate_at_5:.4f}")
    print(f"  Hit Rate@10: {result.hit_rate_at_10:.4f}")
    print(f"  Avg Embed:   {result.avg_embed_ms:.1f}ms")
    print(f"  Avg Search:  {result.avg_vector_search_ms:.1f}ms")
    print(f"  Avg Rerank:  {result.avg_rerank_ms:.1f}ms")


def _print_qa_summary(result) -> None:
    print(f"\n问答评估结果:")
    print(f"  BLEU-1:          {result.avg_bleu_1:.4f}")
    print(f"  BLEU-4:          {result.avg_bleu_4:.4f}")
    print(f"  ROUGE-L F:       {result.avg_rouge_l_f:.4f}")
    print(f"  Faithfulness:    {result.avg_faithfulness:.2f} / 5")
    print(f"  Relevance:       {result.avg_relevance:.2f} / 5")
    print(f"  Completeness:    {result.avg_completeness:.2f} / 5")
    print(f"  Accuracy:        {result.avg_accuracy:.2f} / 5")
    print(f"  Overall:         {result.avg_overall:.2f} / 5")


if __name__ == "__main__":
    app()
