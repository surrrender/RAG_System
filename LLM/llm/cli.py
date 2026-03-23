from __future__ import annotations

import json

import typer

from llm.api import create_app
from llm.config import load_settings
from llm.service import answer_question


app = typer.Typer(help="Local-model RAG QA over the indexed WeChat docs.")


@app.command()
def ask(
    question: str,
    top_k: int | None = typer.Option(None, min=1, help="Number of chunks to retrieve before generation."),
) -> None:
    result = answer_question(question=question, top_k=top_k)
    typer.echo(result.answer)
    typer.echo("")
    typer.echo("Citations:")
    typer.echo(json.dumps([item.to_dict() for item in result.citations], ensure_ascii=False, indent=2))


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
