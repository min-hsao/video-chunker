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
from .analyzer import (
    ChunkInfo, analyze_chunks,
    validate_split_points, detect_retakes_llm, detect_fillers_llm, suggest_content_cuts,
)
from .splitter import (
    MIN_CHUNK_DURATION, build_chunks, clean_video, compute_split_points, split_video,
    compute_clean_segments_llm,
)
from .transcriber import transcribe_audio
from .transcriber import Transcript
from .utils import SilenceInterval, detect_silence, get_keyframes, get_video_info, qc_file, QCResult, generate_contact_sheet

console = Console()

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def _resolve_llm_model(llm_model: str) -> str:
    """Strip 'local/' prefix from llm_model if present (LM Studio shorthand)."""
    return llm_model[len("local/"):] if llm_model.startswith("local/") else llm_model


def _make_llm_client(llm_model: str) -> OpenAI:
    """Build the right OpenAI-compatible client for the given model string."""
    if llm_model.startswith("deepseek"):
        return OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    if llm_model.startswith("local/"):
        return OpenAI(
            api_key="lm-studio",
            base_url=os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        )
    return OpenAI()


# ── Formatting helpers ─────────────────────────────────────────────────────────


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _format_time_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _find_silence_for_split(
    split_time: float,
    silence_intervals: list[SilenceInterval],
    tolerance: float = 10.0,
) -> SilenceInterval | None:
    """Find the silence interval that corresponds to a split point."""
    best = None
    best_dist = float("inf")
    for interval in silence_intervals:
        dist = min(
            abs(interval.end - split_time),
            abs(interval.start - split_time),
            abs(interval.midpoint - split_time),
        )
        if dist < best_dist and dist <= tolerance:
            best_dist = dist
            best = interval
    return best


# ── Review / plan output ──────────────────────────────────────────────────────


def _print_split_plan(
    chunks: list[ChunkInfo],
    silence_intervals: list[SilenceInterval],
    video_duration: float,
) -> None:
    """Print the split plan to stdout."""
    print("\n=== Split Plan ===")
    for chunk in chunks:
        time_range = f"{_format_time_hms(chunk.start)} \u2192 {_format_time_hms(chunk.end)}"
        duration = f"({int(chunk.duration)}s)"

        if chunk.index == 0:
            reason = "[start]"
        elif chunk.retake:
            reason = "[retake]"
        elif chunk.cue_triggered:
            reason = "[cue keyword]"
        else:
            si = _find_silence_for_split(chunk.start, silence_intervals)
            if si:
                reason = f"[silence {si.duration:.1f}s @ {_format_time_hms(si.start)}]"
            else:
                reason = "[silence]"

        print(f"  Chunk {chunk.index + 1:02d}  {time_range}  {duration:>8}  {reason}")

    total = f"{_format_time_hms(0)} \u2192 {_format_time_hms(video_duration)}"
    print(f"\n  Total: {len(chunks)} chunks from {total}")


def _print_silence_report(
    silence_intervals: list[SilenceInterval],
    silence_threshold: float,
    split_points: list[tuple[float, bool, bool]],
) -> None:
    """Print detailed silence detection report to stdout."""
    print(f"\n=== Silence Report (threshold: {silence_threshold} dB) ===")
    print(f"  Detected {len(silence_intervals)} silence intervals:\n")

    split_times = [p for p, _, _ in split_points]

    used_count = 0
    for i, si in enumerate(silence_intervals, 1):
        flags: list[str] = []

        if si.duration < 0.5:
            flags.append("SHORT (<0.5s)")

        # Check if this silence was used as a split point
        used = any(
            abs(si.end - t) <= 2.0 or abs(si.start - t) <= 2.0
            for t in split_times
        )
        if used:
            used_count += 1
        else:
            flags.append(f"discarded (would create chunk < {MIN_CHUNK_DURATION:.0f}s)")

        flag_str = f"  <- {', '.join(flags)}" if flags else ""
        print(
            f"  {i:3d}. {_format_time_hms(si.start)} \u2192 {_format_time_hms(si.end)}"
            f"  ({si.duration:.2f}s){flag_str}"
        )

    discarded_count = len(silence_intervals) - used_count
    if discarded_count:
        print(f"\n  {used_count} used as split points, {discarded_count} discarded")


# ── Auto-tune ─────────────────────────────────────────────────────────────────


def _auto_tune_silence(
    video_path: Path,
    silence_duration: float,
) -> tuple[list[SilenceInterval], float]:
    """Run silence detection at multiple thresholds and let user pick."""
    thresholds = [-25, -30, -35, -40]
    results: list[tuple[float, list[SilenceInterval]]] = []

    print("\n=== Auto-Tune Silence Detection ===")
    print("  Testing thresholds...\n")
    for threshold in thresholds:
        intervals = detect_silence(
            video_path,
            silence_duration=silence_duration,
            silence_threshold=threshold,
        )
        results.append((threshold, intervals))
        marker = "  (default)" if threshold == -30 else ""
        print(
            f"  {threshold:>4} dB \u2192 {len(intervals):>3} silences detected"
            f" \u2192 ~{len(intervals) + 1} chunks{marker}"
        )

    print()
    choice = input("Select threshold (dB) or press Enter for default (-30): ").strip()

    selected_threshold = -30.0
    if choice:
        try:
            selected_threshold = float(choice)
        except ValueError:
            print(f"  Invalid input '{choice}', using default (-30 dB)")

    for threshold, intervals in results:
        if threshold == selected_threshold:
            print(f"  Using {threshold} dB ({len(intervals)} silences)\n")
            return intervals, threshold

    # User entered a threshold not in the preset list — run it
    print(f"  Running detection at {selected_threshold} dB...")
    intervals = detect_silence(
        video_path,
        silence_duration=silence_duration,
        silence_threshold=selected_threshold,
    )
    print(f"  Found {len(intervals)} silences\n")
    return intervals, selected_threshold


# ── CLI ────────────────────────────────────────────────────────────────────────


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_dir", type=click.Path(path_type=Path), default=None, help="Output directory for chunks. Defaults to a 'chunks' folder next to the source video/directory.")
@click.option("--type", "video_type", default="product-demo", help="Video type: product-demo, tutorial, talking-head, or custom string.")
@click.option("--script", type=click.Path(exists=True, path_type=Path), default=None, help="Path to draft script file for comparison.")
@click.option("--cues", default="cut,next,take", help="Comma-separated verbal cue keywords that force a split.")
@click.option("--silence-duration", type=float, default=3.0, help="Minimum silence duration in seconds to consider as a break.")
@click.option("--silence-threshold", type=float, default=-30, help="Silence threshold in dB (default: -30).")
@click.option("--detailed", is_flag=True, help="Output full JSON manifest.")
@click.option("--whisper-mode", default="local", type=click.Choice(["local", "openai"], case_sensitive=False), help="Whisper mode: local (free, runs on your machine) or openai (paid API).")
@click.option("--whisper-model", default="small", help="Whisper model: tiny/base/small/medium/large-v3 for local, or whisper-1 for openai.")
@click.option("--language", default=None, help="Language code for transcription (e.g. 'en'). Auto-detect if not set.")
@click.option("--llm-model", default="deepseek-chat", help="LLM model for chunk analysis. Use 'deepseek-chat' for DeepSeek, or 'gpt-4o' for OpenAI.")
@click.option("--dry-run", is_flag=True, help="Show detected chunks without splitting.")
@click.option("--review", is_flag=True, help="Show split plan and ask for confirmation before splitting.")
@click.option("--auto-tune", is_flag=True, help="Test multiple silence thresholds and let you choose.")
@click.option("--silence-report", is_flag=True, help="Show detailed silence detection report.")
@click.option("--clean", is_flag=True, help="Clean mode: remove filler words, silences, and retakes. Outputs a single stitched file instead of chunks.")
@click.option("--smart-splits", is_flag=True, help="Use LLM to validate split points — only split at natural topic boundaries.")
@click.option("--smart-retakes", is_flag=True, help="Use LLM for retake detection — catches verbal cues, semantic retakes, abandoned thoughts.")
@click.option("--smart-fillers", is_flag=True, help="Use LLM for filler detection — understands context instead of hardcoded word list.")
@click.option("--smart-clean", is_flag=True, help="Use LLM for content-aware clean mode — removes tangents, redundancy, off-topic chatter. (Requires --clean)")
@click.option("--contact-sheet", is_flag=True, help="Generate a thumbnail contact sheet (JPEG) for each chunk after splitting.")
@click.option("--qc", is_flag=True, help="Run ffprobe quality control on each output file after splitting.")
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
    language: str | None,
    llm_model: str,
    dry_run: bool,
    review: bool,
    auto_tune: bool,
    silence_report: bool,
    clean: bool,
    smart_splits: bool,
    smart_retakes: bool,
    smart_fillers: bool,
    smart_clean: bool,
    contact_sheet: bool,
    qc: bool,
    verbose: bool,
) -> None:
    """Split a long video recording into labeled chunks.

    INPUT_PATH can be a single video file (MP4, MOV, etc.) or a directory
    to batch-process all videos inside it.
    """
    setup_logging(verbose)

    # Default output: "chunks" folder next to the source video/directory
    if output_dir is None:
        if clean:
            output_dir = input_path.parent / "cleaned"
        else:
            output_dir = input_path.parent / "chunks"

    script_text: str | None = None
    if script:
        script_text = script.read_text(encoding="utf-8")
        console.print(f"[dim]Loaded script from {script}[/dim]")

    # Build shared clients once (reused across batch)
    whisper_client = OpenAI() if whisper_mode == "openai" else None
    effective_llm_model = _resolve_llm_model(llm_model)
    llm_client = _make_llm_client(llm_model)

    if input_path.is_dir():
        # ── Batch mode ──────────────────────────────────────────────────────
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
                    language=language,
                    llm_model=effective_llm_model,
                    dry_run=dry_run,
                    review=review,
                    auto_tune=auto_tune,
                    silence_report=silence_report,
                    clean=clean,
                    smart_splits=smart_splits,
                    smart_retakes=smart_retakes,
                    smart_fillers=smart_fillers,
                    smart_clean=smart_clean,
                    contact_sheet=contact_sheet,
                    qc=qc,
                    whisper_client=whisper_client,
                    llm_client=llm_client,
                )
            except Exception as e:
                console.print(f"[red]Failed:[/red] {video.name} \u2014 {e}")
                failed.append(video.name)

        console.rule()
        if failed:
            console.print(f"\n[yellow]Completed with {len(failed)} failure(s):[/yellow]")
            for name in failed:
                console.print(f"  [red]\u2717[/red] {name}")
        else:
            console.print(f"\n[green bold]All {len(videos)} videos processed successfully![/green bold]")
            console.print(f"Chunks saved under {output_dir}/")

    else:
        # ── Single file mode ─────────────────────────────────────────────────
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
            language=language,
            llm_model=effective_llm_model,
            dry_run=dry_run,
            review=review,
            auto_tune=auto_tune,
            silence_report=silence_report,
            clean=clean,
            smart_splits=smart_splits,
            smart_retakes=smart_retakes,
            smart_fillers=smart_fillers,
            smart_clean=smart_clean,
            contact_sheet=contact_sheet,
            qc=qc,
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
    language: str | None,
    llm_model: str,
    dry_run: bool,
    review: bool,
    auto_tune: bool,
    silence_report: bool,
    clean: bool,
    smart_splits: bool,
    smart_retakes: bool,
    smart_fillers: bool,
    smart_clean: bool,
    contact_sheet: bool,
    qc: bool,
    whisper_client: OpenAI | None,
    llm_client: OpenAI,
) -> None:
    """Process a single video file end-to-end."""
    logger = logging.getLogger(__name__)
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

        # Step 2: Detect silence (or auto-tune)
        if auto_tune:
            progress.stop()
            silence_intervals, silence_threshold = _auto_tune_silence(
                input_video, silence_duration
            )
            progress.start()
        else:
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

        # Step 3: Transcribe
        task = progress.add_task(
            f"Transcribing ({whisper_mode} / {whisper_model})...", total=None
        )
        try:
            transcript = transcribe_audio(
                input_video,
                model=whisper_model,
                mode=whisper_mode,
                language=language,
                client=whisper_client,
            )
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e
        console.print(f"[dim]Transcript: {len(transcript.text)} chars, {len(transcript.segments)} segments[/dim]")
        if transcript.words:
            console.print(f"[dim]Word-level timestamps: {len(transcript.words)} words[/dim]")
        progress.update(task, completed=1, total=1)

    # ── Clean mode branch ──────────────────────────────────────────────────
    if clean:
        _process_clean(
            input_video, output_dir, info,
            transcript=transcript,
            silence_intervals=silence_intervals,
            llm_client=llm_client,
            llm_model=llm_model,
            smart_retakes=smart_retakes,
            smart_fillers=smart_fillers,
            smart_clean=smart_clean,
            dry_run=dry_run,
        )
        return

    # ── Chunk mode ───────────────────────────────────────────────────────────

    # Step 4: Keyframes
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Finding keyframes...", total=None)
        try:
            keyframes = get_keyframes(input_video)
        except RuntimeError:
            logger.warning("Could not extract keyframes \u2014 splits may not be frame-accurate")
            keyframes = []
        progress.update(task, completed=1, total=1)

    # Step 5: Compute split points
    split_points = compute_split_points(
        silence_intervals,
        transcript,
        info.duration,
        cue_keywords=cue_keywords,
        keyframes=keyframes if keyframes else None,
    )
    console.print(f"[dim]Computed {len(split_points)} split points[/dim]")

    # Step 5b: Smart retake detection via LLM
    if smart_retakes:
        console.print("[bold cyan]Smart retakes:[/bold cyan] detecting with LLM...")
        retake_detections = detect_retakes_llm(
            transcript.segments,
            model=llm_model,
            client=llm_client,
        )
        retake_times = [r.timestamp for r in retake_detections if r.is_retake]
        if retake_times:
            console.print(f"[dim]  LLM found {len(retake_times)} additional retakes[/dim]")
            from .splitter import _safe_split_point
            for rt in retake_times:
                safe = _safe_split_point(rt, transcript, silence_intervals)
                split_points.append((safe, True, True))
            # Re-sort and deduplicate
            split_points.sort(key=lambda x: x[0])
            deduped = [split_points[0]] if split_points else []
            for pt, cue, ret in split_points[1:]:
                if pt - deduped[-1][0] >= 1.0:
                    deduped.append((pt, cue, ret))
                elif (cue or ret) and not (deduped[-1][1] or deduped[-1][2]):
                    deduped[-1] = (pt, cue, ret)
            split_points = deduped
            console.print(f"[dim]  Now {len(split_points)} split points after retakes[/dim]")

    # Step 5c: Smart split point validation via LLM
    if smart_splits and split_points:
        console.print("[bold cyan]Smart splits:[/bold cyan] validating split points with LLM...")
        candidate_times = [p for p, _, _ in split_points]
        validations = validate_split_points(
            candidate_times,
            transcript_text=transcript.text,
            segments=transcript.segments,
            model=llm_model,
            client=llm_client,
        )
        approved = [v.timestamp for v in validations if v.approved]
        rejected = [v for v in validations if not v.approved]
        # Keep cue/retake splits regardless of LLM opinion
        preserved = [(p, c, r) for p, c, r in split_points if c or r]
        validated_silence = [(v, False, False) for v in approved]
        split_points = preserved + validated_silence
        split_points.sort(key=lambda x: x[0])
        if rejected:
            console.print(f"[dim]  Rejected {len(rejected)} mid-thought pauses:[/dim]")
            for v in rejected:
                console.print(f"[dim]    {v.timestamp:.1f}s: {v.reason}[/dim]")
        console.print(f"[dim]  Kept {len(split_points)} validated split points[/dim]")

    # Step 6: Build chunks
    chunks = build_chunks(split_points, transcript, info.duration)
    console.print(f"\n[bold]Detected {len(chunks)} chunks[/bold]")

    # ── Review mode ──────────────────────────────────────────────────────────
    if review:
        _print_split_plan(chunks, silence_intervals, info.duration)

        if silence_report:
            _print_silence_report(silence_intervals, silence_threshold, split_points)

        if dry_run:
            print("\nDry run \u2014 no files written.")
            return

        print()
        answer = input("Proceed with splitting? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    # ── Silence report (without review) ──────────────────────────────────────
    if silence_report and not review:
        _print_silence_report(silence_intervals, silence_threshold, split_points)

    # Step 7: LLM analysis
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing chunks...", total=len(chunks))
        try:
            chunks = analyze_chunks(
                chunks,
                video_type=video_type,
                script=script_text,
                model=llm_model,
                client=llm_client,
            )
        except Exception as e:
            raise RuntimeError(f"LLM analysis failed: {e}") from e
        progress.update(task, completed=len(chunks))

    _print_chunks_table(chunks)

    if dry_run:
        console.print("\n[yellow]Dry run \u2014 no files written.[/yellow]")
        if detailed:
            _print_manifest(chunks, input_video, info)
        return

    # Step 8: Split
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

    # ── Contact sheet generation ─────────────────────────────────────────────
    if contact_sheet:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating contact sheets...", total=len(chunks))
            for chunk in chunks:
                if chunk.output_path and Path(chunk.output_path).exists():
                    sheet_path = Path(chunk.output_path).with_suffix(".jpg")
                    try:
                        generate_contact_sheet(
                            Path(chunk.output_path),
                            sheet_path,
                            num_thumbnails=4,
                            grid_cols=2,
                        )
                        chunk.contact_sheet_path = str(sheet_path)
                    except Exception as e:
                        console.print(f"[yellow]Contact sheet failed for chunk {chunk.index + 1}:[/yellow] {e}")
                progress.update(task, advance=1)
        console.print("[dim]Contact sheets generated[/dim]")

    # ── QC verification ─────────────────────────────────────────────────────────
    if qc:
        console.print("\n[bold]QC Report:[/bold]")
        qc_table = Table(title="Quality Control", show_lines=True)
        qc_table.add_column("#", style="bold", width=4)
        qc_table.add_column("File", width=30)
        qc_table.add_column("Duration", width=14)
        qc_table.add_column("Resolution", width=12)
        qc_table.add_column("A/V", width=6)
        qc_table.add_column("Size", width=10)
        qc_table.add_column("Status", width=10)

        all_pass = True
        for chunk in chunks:
            if not chunk.output_path:
                continue
            result = qc_file(
                Path(chunk.output_path),
                expected_duration=chunk.duration,
            )
            chunk.qc_result = result

            status = "[green]\u2713 pass[/green]" if not result.errors else "[red]\u2717 fail[/red]"
            if result.errors:
                all_pass = False

            av = f"{'V' if result.has_video else '-'}{'A' if result.has_audio else '-'}"
            res = f"{result.width}x{result.height}" if result.has_video else "-"
            fname = Path(chunk.output_path).name

            qc_table.add_row(
                str(chunk.index + 1),
                fname,
                f"{result.duration:.1f}s / {result.expected_duration:.1f}s",
                res,
                av,
                f"{result.file_size_mb:.1f} MB",
                status,
            )
            if result.errors:
                for err in result.errors:
                    qc_table.add_row("", "", "", "", "", "", f"[red]{err}[/red]")

        console.print(qc_table)
        if all_pass:
            console.print("[green bold]All chunks passed QC.[/green bold]")
        else:
            console.print("[red bold]Some chunks failed QC — check errors above.[/red bold]")

    if detailed:
        _print_manifest(chunks, input_video, info)


def _process_clean(
    input_video: Path,
    output_dir: Path,
    info,
    *,
    transcript: Transcript,
    silence_intervals: list,
    llm_client: OpenAI,
    llm_model: str,
    smart_retakes: bool = False,
    smart_fillers: bool = False,
    smart_clean: bool = False,
    dry_run: bool = False,
) -> None:
    """Clean mode: remove filler words, silences, retakes and stitch."""
    from .splitter import compute_clean_segments

    llm_retakes = None
    llm_fillers = None
    llm_content_cuts = None

    # Run LLM analyses if requested
    if smart_retakes or smart_fillers or smart_clean:
        console.print("[bold cyan]Smart clean:[/bold cyan] running LLM analysis...")

    if smart_retakes:
        console.print("  [dim]Detecting retakes with LLM...[/dim]")
        llm_retakes = detect_retakes_llm(
            transcript.segments,
            model=llm_model,
            client=llm_client,
        )
        retake_count = sum(1 for r in llm_retakes if r.is_retake)
        console.print(f"  [dim]  Found {retake_count} retakes[/dim]")

    if smart_fillers:
        console.print("  [dim]Detecting fillers with LLM...[/dim]")
        llm_fillers = detect_fillers_llm(
            transcript.segments,
            model=llm_model,
            client=llm_client,
        )
        filler_count = sum(1 for f in llm_fillers if f.is_filler)
        console.print(f"  [dim]  Found {filler_count} filler segments[/dim]")

    if smart_clean:
        console.print("  [dim]Identifying tangents/redundancy with LLM...[/dim]")
        llm_content_cuts = suggest_content_cuts(
            transcript.segments,
            model=llm_model,
            client=llm_client,
        )
        console.print(f"  [dim]  Found {len(llm_content_cuts)} content cut suggestions[/dim]")

    # Use LLM-enhanced or basic clean segments
    if llm_retakes or llm_fillers or llm_content_cuts:
        keep = compute_clean_segments_llm(
            transcript,
            silence_intervals,
            info.duration,
            llm_retakes=llm_retakes,
            llm_fillers=llm_fillers,
            llm_content_cuts=llm_content_cuts,
        )
    else:
        keep = compute_clean_segments(
            transcript,
            silence_intervals,
            info.duration,
        )

    original = info.duration
    kept = sum(s.end - s.start for s in keep)
    cut = original - kept
    pct = (cut / original * 100) if original > 0 else 0

    console.print(f"\n[bold]Clean Mode[/bold]")
    console.print(f"  Original:  {_format_time_hms(original)} ({original:.1f}s)")
    console.print(f"  Kept:      {_format_time_hms(kept)} ({kept:.1f}s)")
    console.print(f"  Cut:       {_format_time_hms(cut)} ({cut:.1f}s / {pct:.0f}%)")
    console.print(f"  Segments:  {len(keep)} kept, {len(keep) - 1} cuts")

    # Show what's being cut
    if cut > 0:
        console.print("\n[bold]Segments being kept:[/bold]")
        for i, seg in enumerate(keep, 1):
            console.print(
                f"  {i:2d}. {_format_time_hms(seg.start)} \u2192 {_format_time_hms(seg.end)}"
                f"  ({seg.end - seg.start:.1f}s)"
            )

    if dry_run:
        console.print("\n[yellow]Dry run \u2014 no files written.[/yellow]")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_video.stem}_clean.mp4"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Cleaning video...", total=None)
        result = clean_video(
            input_video,
            transcript,
            silence_intervals,
            output_path,
        )
        progress.update(task, completed=1, total=1)

    console.print(
        f"\n[green bold]Done![/green bold] Cleaned video saved to {output_path}"
        f"\n  {result['original_duration']:.1f}s \u2192 {result['clean_duration']:.1f}s"
        f" ({result['cut_duration']:.1f}s removed, {result['cuts_made']} cuts)"
    )


def _print_chunks_table(chunks: list[ChunkInfo]) -> None:
    table = Table(title="Detected Chunks", show_lines=True)
    table.add_column("#", style="bold", width=4)
    table.add_column("Time Range", width=16)
    table.add_column("Duration", width=10)
    table.add_column("Description", width=30)
    table.add_column("Status", width=12)
    table.add_column("Split", width=8)

    for chunk in chunks:
        time_range = f"{_format_time(chunk.start)} - {_format_time(chunk.end)}"
        duration_str = f"{chunk.duration:.1f}s"

        if chunk.analysis:
            desc = chunk.analysis.description
            status = "[green]complete[/green]" if chunk.analysis.is_complete else "[yellow]incomplete[/yellow]"
        else:
            desc = "\u2014"
            status = "\u2014"

        if chunk.retake:
            split_reason = "[magenta]retake[/magenta]"
        elif chunk.cue_triggered:
            split_reason = "[cyan]cue[/cyan]"
        else:
            split_reason = "[dim]silence[/dim]"

        table.add_row(str(chunk.index + 1), time_range, duration_str, desc, status, split_reason)

    console.print(table)


def _print_manifest(chunks: list[ChunkInfo], input_video: Path, info) -> None:
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
        chunk_data: dict = {
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
        if chunk.contact_sheet_path:
            chunk_data["contact_sheet"] = chunk.contact_sheet_path
        if chunk.qc_result:
            r = chunk.qc_result
            chunk_data["qc"] = {
                "passed": len(r.errors) == 0,
                "duration": r.duration,
                "expected_duration": r.expected_duration,
                "resolution": f"{r.width}x{r.height}",
                "has_video": r.has_video,
                "has_audio": r.has_audio,
                "file_size_mb": round(r.file_size_mb, 2),
                "errors": r.errors,
            }
        manifest["chunks"].append(chunk_data)

    console.print("\n[bold]Manifest:[/bold]")
    console.print_json(json.dumps(manifest, indent=2))
