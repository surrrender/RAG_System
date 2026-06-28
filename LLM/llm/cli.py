from __future__ import annotations

import json

import typer

from llm.api import create_app
from llm.config import load_settings
from llm.service import build_service


app = typer.Typer(help="Local-model RAG QA over the indexed WeChat docs.")


@app.command()
def ask(
    question: str,
    top_k: int | None = typer.Option(None, min=1, help="Number of chunks to retrieve before generation."),
    provider: str = typer.Option(
        "ollama", help="Generation provider: 'ollama' (local) or 'deepseek' (API)."
    ),
) -> None:
    settings = load_settings()
    settings.generation_provider = provider
    service = build_service(settings)
    citations: list[object] = []

    for event in service.stream_answer_question(question=question, top_k=top_k or settings.top_k):
        if event["event"] == "delta":
            typer.echo(str(event["data"].get("text") or ""), nl=False)
        elif event["event"] == "citations":
            citations = list(event["data"].get("citations") or [])

    typer.echo("")
    typer.echo("Citations:")
    typer.echo(json.dumps(citations, ensure_ascii=False, indent=2))


@app.command()
def serve() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required to serve the HTTP API. Install the LLM project dependencies first."
        ) from exc

    settings = load_settings()
    app_instance = create_app(settings=settings)
    uvicorn.run(app_instance, host=settings.http_host, port=settings.http_port)
