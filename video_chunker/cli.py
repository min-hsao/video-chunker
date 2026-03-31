"""CLI entry point for video-chunker."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from current dir or any parent — silent if not found
load_dotenv()
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__
from .analyzer import ChunkInfo, analyze_chunks
from .splitter import build_chunks, compute_split_points, split_video
from .transcriber import transcribe_audio
from .utils import detect_silence, get_keyframes, get_video_info

console = Console()

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def make_clients(whisper_mode: str, llm_model: str):
    """Create and return (whisper_client, llm_client)."""
    whisper_client = OpenAI() if whisper_mode == "openai" else None

    if llm_model.startswith("deepseek"):
        llm_client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    elif llm_model.startswith("local/"):
        llm_client = OpenAI(
            api_key="lm-studio",
            base_url=os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        )
    else:
        llm_client = OpenAI()

    return whisper_client, llm_client


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_dir", type=click.Path(path_type=Path), default="./chunks", help="Output directory for chunks.")
@click.option("--type", "video_type", default="product-demo", help="Video type: product-demo, tutorial, talking-head, or custom string.")
@click.option("--script", type=click.Path(exists=True, path_type=Path), default=None, help="Path to draft script file for comparison.")
@click.option("--cues", default="cut,next,take", help="Comma-separated verbal cue keywords that force a split.")
@click.option("--silence-duration", type=float, default=2.0, help="Minimum silence duration in seconds to consider as a break.")
@click.option("--silence-threshold", type=float, default=-35, help="Silence threshold in dB.")
@click.option("--detailed", is_flag=True, help="Output full JSON manifest.")
@click.option("--whisper-mode", default="local", type=click.Choice(["local", "openai"], case_sensitive=False), help="Whisper mode: local (free, runs on your machine) or openai (paid API).")
@click.option("--whisper-model", default="base", help="Whisper model: tiny/base/small/medium/large-v3 for local, or whisper-1 for openai.")
@click.option("--llm-model", default="deepseek-chat", help="LLM model for chunk analysis. Use 'deepseek-chat' for DeepSeek, or 'gpt-4o' for OpenAI.")
@click.option("--dry-run", is_flag=True, help="Show detected chunks without splitting.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose/debug logging.")
@click.version_option(version=__version__)
def cli(
    input_path: Path,
    output_dir: Path,
    video_type: str,
    script: Path | None,
    cues: str,
    silence_duration: float,
    silence_threshold: float,
    detailed: bool,
    whisper_mode: str,
    whisper_model: str,
    llm_model: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Split a long video recording into labeled chunks.

    INPUT_PATH can be a single video file (MP4, MOV, etc.) or a directory
    to batch-process all videos inside it.
    """
    setup_logging(verbose)

    # Read script if provided
    script_text: str | None = None
    if script:
        script_text = script.read_text(encoding="utf-8")
        console.print(f"[dim]Loaded script from {script}[/dim]")

    # Shared options dict to pass to process_video
    opts = dict(
        output_dir=output_dir,
        video_type=video_type,
        script_text=script_text,
        cues=cues,
        silence_duration=silence_duration,
        silence_threshold=silence_threshold,
        detailed=detailed,
        whisper_mode=whisper_mode,
        whisper_model=whisper_model,
        llm_model=llm_model,
        dry_run=dry_run,
    )

    # Build clients once (reuse across batch)
    # Strip "local/" prefix for llm_model if needed before building clients
    effective_llm_model = llm_model[len("local/"):] if llm_model.startswith("local/") else llm_model
    whisper_client, llm_client = make_clients(whisper_mode, effective_llm_model)

    if input_path.is_dir():
        # Batch mode
        videos = sorted([
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ])

        if not videos:
            console.print(f"[red]No video files found in {input_path}[/red]")
            sys.exit(1)

        console.print(f"\n[bold]Batch mode:[/bold] found {len(videos)} video(s) in {input_path}\n")

        failed = []
        for i, video in enumerate(videos, 1):
            console.rule(f"[bold cyan]{i}/{len(videos)}: {video.name}[/bold cyan]")
            # Each video gets its own subfolder
            video_output = output_dir / video.stem
            try:
                process_video(
                    video,
                    output_dir=video_output,
                    video_type=video_type,
                    script_text=script_text,
                    cues=cues,
                    silence_duration=silence_duration,
                    silence_threshold=silence_threshold,
                    detailed=detailed,
                    whisper_mode=whisper_mode,
                    whisper_model=whisper_model,
                    llm_model=llm_model,
                    dry_run=dry_run,
                    whisper_client=whisper_client,
                    llm_client=llm_client,
                )
            except Exception as e:
                console.print(f"[red]Failed:[/red] {video.name} — {e}")
                failed.append(video.name)

        console.rule()
        if failed:
            console.print(f"\n[yellow]Completed with {len(failed)} failure(s):[/yellow]")
            for f in failed:
                console.print(f"  [red]✗[/red] {f}")
        else:
            console.print(f"\n[green bold]All {len(videos)} videos processed successfully![/green bold]")
            console.print(f"Chunks saved to {output_dir}/")
    else:
        # Single file mode
        process_video(
            input_path,
            output_dir=output_dir,
            video_type=video_type,
            script_text=script_text,
            cues=cues,
            silence_duration=silence_duration,
            silence_threshold=silence_threshold,
            detailed=detailed,
            whisper_mode=whisper_mode,
            whisper_model=whisper_model,
            llm_model=llm_model,
            dry_run=dry_run,
            whisper_client=whisper_client,
            llm_client=llm_client,
        )


def process_video(
    input_video: Path,
    *,
    output_dir: Path,
    video_type: str,
    script_text: str | None,
    cues: str,
    silence_duration: float,
    silence_threshold: float,
    detailed: bool,
    whisper_mode: str,
    whisper_model: str,
    llm_model: str,
    dry_run: bool,
    whisper_client,
    llm_client,
) -> None:
    """Process a single video file."""
    logger = logging.getLogger(__name__)

    # Strip "local/" prefix for llm_model if needed
    effective_llm_model = llm_model[len("local/"):] if llm_model.startswith("local/") else llm_model

    cue_keywords = [k.strip() for k in cues.split(",") if k.strip()]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Step 1: Probe video
        task = progress.add_task("Probing video...", total=None)
        try:
            info = get_video_info(input_video)
        except RuntimeError as e:
            raise RuntimeError(f"Probe failed: {e}") from e

        console.print(
            f"[bold]Input:[/bold] {input_video.name} | "
            f"{info.codec} {info.width}x{info.height} | "
            f"{info.duration:.1f}s | {info.fps:.1f}fps"
        )
        progress.update(task, completed=1, total=1)

        # Step 2: Detect silence
        task = progress.add_task("Detecting silence...", total=None)
        try:
            silence_intervals = detect_silence(
                input_video,
                silence_duration=silence_duration,
                silence_threshold=silence_threshold,
            )
        except RuntimeError as e:
            raise RuntimeError(f"Silence detection failed: {e}") from e

        console.print(f"[dim]Found {len(silence_intervals)} silence intervals[/dim]")
        progress.update(task, completed=1, total=1)

        # Step 3: Transcribe audio
        task = progress.add_task("Transcribing audio...", total=None)
        try:
            transcript = transcribe_audio(
                input_video,
                model=whisper_model,
                mode=whisper_mode,
                client=whisper_client,
            )
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e

        console.print(f"[dim]Transcript: {len(transcript.text)} chars, {len(transcript.segments)} segments[/dim]")
        progress.update(task, completed=1, total=1)

        # Step 4: Get keyframes
        task = progress.add_task("Finding keyframes...", total=None)
        try:
            keyframes = get_keyframes(input_video)
        except RuntimeError:
            logger.warning("Could not extract keyframes, splits may not be frame-accurate")
            keyframes = []
        progress.update(task, completed=1, total=1)

        # Step 5: Compute split points
        task = progress.add_task("Computing split points...", total=None)
        split_points = compute_split_points(
            silence_intervals,
            transcript,
            info.duration,
            cue_keywords=cue_keywords,
            keyframes=keyframes if keyframes else None,
        )
        console.print(f"[dim]Computed {len(split_points)} split points[/dim]")
        progress.update(task, completed=1, total=1)

        # Step 6: Build chunks
        chunks = build_chunks(split_points, transcript, info.duration)
        console.print(f"\n[bold]Detected {len(chunks)} chunks[/bold]")

        # Step 7: Analyze chunks with LLM
        task = progress.add_task("Analyzing chunks...", total=len(chunks))
        try:
            chunks = analyze_chunks(
                chunks,
                video_type=video_type,
                script=script_text,
                model=effective_llm_model,
                client=llm_client,
            )
        except Exception as e:
            raise RuntimeError(f"LLM analysis failed: {e}") from e
        progress.update(task, completed=len(chunks))

    # Display results table
    _print_chunks_table(chunks)

    if dry_run:
        console.print("\n[yellow]Dry run — no files written.[/yellow]")
        if detailed:
            _print_manifest(chunks, input_video, info)
        return

    # Step 8: Split video
    extension = info.container if info.container in ("mp4", "mov", "mkv") else "mp4"
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Splitting video...", total=len(chunks))
        try:
            chunks = split_video(input_video, chunks, output_dir, extension=extension)
        except RuntimeError as e:
            raise RuntimeError(f"Split failed: {e}") from e
        progress.update(task, completed=len(chunks))

    console.print(f"\n[green bold]Done![/green bold] {len(chunks)} chunks saved to {output_dir}/")

    if detailed:
        _print_manifest(chunks, input_video, info)


def _print_chunks_table(chunks: list[ChunkInfo]) -> None:
    """Print a rich table of detected chunks."""
    table = Table(title="Detected Chunks", show_lines=True)
    table.add_column("#", style="bold", width=4)
    table.add_column("Time Range", width=16)
    table.add_column("Duration", width=10)
    table.add_column("Description", width=30)
    table.add_column("Status", width=12)
    table.add_column("Cue?", width=5)

    for chunk in chunks:
        start_str = _format_time(chunk.start)
        end_str = _format_time(chunk.end)
        time_range = f"{start_str} - {end_str}"
        duration_str = f"{chunk.duration:.1f}s"

        if chunk.analysis:
            desc = chunk.analysis.description
            status = "[green]complete[/green]" if chunk.analysis.is_complete else "[yellow]incomplete[/yellow]"
        else:
            desc = "—"
            status = "—"

        cue = "[cyan]yes[/cyan]" if chunk.cue_triggered else ""

        table.add_row(
            str(chunk.index + 1),
            time_range,
            duration_str,
            desc,
            status,
            cue,
        )

    console.print(table)


def _print_manifest(chunks: list[ChunkInfo], input_video: Path, info) -> None:
    """Print JSON manifest to stdout."""
    manifest = {
        "input": str(input_video),
        "video_info": {
            "codec": info.codec,
            "resolution": f"{info.width}x{info.height}",
            "duration": info.duration,
            "fps": info.fps,
        },
        "chunks": [],
    }

    for chunk in chunks:
        chunk_data = {
            "index": chunk.index + 1,
            "start": chunk.start,
            "end": chunk.end,
            "duration": chunk.duration,
            "transcript": chunk.transcript,
            "cue_triggered": chunk.cue_triggered,
            "output_path": chunk.output_path,
        }
        if chunk.analysis:
            chunk_data["analysis"] = {
                "description": chunk.analysis.description,
                "is_complete": chunk.analysis.is_complete,
                "confidence": chunk.analysis.confidence,
                "notes": chunk.analysis.notes,
                "script_match": chunk.analysis.script_match,
            }
        manifest["chunks"].append(chunk_data)

    console.print("\n[bold]Manifest:[/bold]")
    console.print_json(json.dumps(manifest, indent=2))


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
