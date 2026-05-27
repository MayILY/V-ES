from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .pipeline import inspect_media, run_pipeline

app = typer.Typer(
    help="Local-first video evidence extraction and summarization CLI.",
    no_args_is_help=True,
)


@app.command()
def inspect(
    input_file: Path = typer.Argument(..., exists=True, readable=True, help="Local media file to inspect."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional config.yaml path."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing generated files."),
) -> None:
    """Run ffprobe and write metadata.json."""
    cfg = load_config(config)
    result = inspect_media(input_file=input_file, output_dir=output, config=cfg, force=force)
    typer.echo(f"metadata: {result}")


@app.command()
def run(
    input_file: Path = typer.Argument(..., exists=True, readable=True, help="Local media file to process."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional config.yaml path."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing generated files."),
    frame_interval: Optional[float] = typer.Option(None, "--frame-interval", help="Seconds between sampled frames."),
    skip_summary: bool = typer.Option(False, "--skip-summary", help="Skip OpenAI summary generation."),
) -> None:
    """Run the MVP pipeline for one local media file."""
    cfg = load_config(config)
    if frame_interval is not None:
        cfg.ffmpeg.frame_interval_sec = frame_interval
    outputs = run_pipeline(
        input_file=input_file,
        output_dir=output,
        config=cfg,
        force=force,
        skip_summary=skip_summary,
    )
    typer.echo("pipeline outputs:")
    for name, path in outputs.items():
        typer.echo(f"- {name}: {path}")
