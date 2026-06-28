"""Audio transcription using local Whisper or OpenAI Whisper API."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .utils import extract_audio

logger = logging.getLogger(__name__)

# Whisper API file size limit (25 MB)
MAX_FILE_SIZE = 25 * 1024 * 1024


@dataclass
class WordSegment:
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    text: str
    segments: list[TranscriptSegment]
    words: list[WordSegment] | None = None


def transcribe_audio(
    video_path: Path,
    *,
    model: str = "base",
    mode: str = "local",
    language: str | None = None,
    client: OpenAI | None = None,
) -> Transcript:
    """Transcribe audio from a video file.

    Args:
        video_path: Path to the video file
        model: Whisper model name. For local: tiny/base/small/medium/large-v3.
               For openai: whisper-1.
        mode: "local" (default, free) or "openai" (paid API)
        language: Optional language code (e.g. "en"). Auto-detect if None.
        client: OpenAI client (required for openai mode)
    """
    logger.info("Extracting audio from %s", video_path.name)
    audio_path = extract_audio(video_path)

    try:
        if mode == "local":
            return _transcribe_local(audio_path, model=model, language=language)
        else:
            if client is None:
                client = OpenAI()
            return _transcribe_openai(audio_path, model=model, client=client, language=language)
    finally:
        audio_path.unlink(missing_ok=True)


def _transcribe_local(
    audio_path: Path,
    *,
    model: str = "base",
    language: str | None = None,
) -> Transcript:
    """Transcribe using local Whisper library (free, runs on-device)."""
    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )

    # Pick the best available device: CUDA > CPU
    # NOTE: MPS (Apple Silicon Metal) is intentionally skipped — Whisper's MPS
    # support silently returns empty transcripts due to fp64 limitations.
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logger.info("Transcribing with local Whisper model '%s' on %s", model, device)

    whisper_model = whisper.load_model(model, device=device)

    kwargs: dict = {
        "word_timestamps": True,
        "fp16": device == "cuda",  # fp16 only on CUDA; CPU uses fp32
    }
    if language:
        kwargs["language"] = language

    result = whisper_model.transcribe(str(audio_path), **kwargs)

    # Sanity check — if transcript is empty, something went wrong
    if not result.get("text", "").strip():
        logger.warning("Whisper returned empty transcript — audio may be silent or too short")

    segments = []
    all_words: list[WordSegment] = []

    for seg in result.get("segments", []):
        segments.append(TranscriptSegment(
            text=seg.get("text", "").strip(),
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
        ))
        # Extract word-level timestamps if available
        for w in seg.get("words", []):
            all_words.append(WordSegment(
                word=w.get("word", "").strip(),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
            ))

    return Transcript(
        text=result.get("text", "").strip(),
        segments=segments,
        words=all_words if all_words else None,
    )


def _transcribe_openai(
    audio_path: Path,
    *,
    model: str,
    client: OpenAI,
    language: str | None = None,
) -> Transcript:
    """Transcribe using OpenAI Whisper API."""
    logger.info("Transcribing with OpenAI Whisper API model '%s'", model)

    file_size = audio_path.stat().st_size
    if file_size <= MAX_FILE_SIZE:
        return _transcribe_openai_file(audio_path, model=model, client=client, language=language)

    logger.info("Audio is %.1f MB — splitting into chunks", file_size / 1024 / 1024)
    return _transcribe_openai_large_file(audio_path, model=model, client=client, language=language)


def _transcribe_openai_file(
    audio_path: Path,
    *,
    model: str,
    client: OpenAI,
    language: str | None = None,
) -> Transcript:
    """Transcribe a single audio file via OpenAI API."""
    kwargs: dict = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"],
    }
    if language:
        kwargs["language"] = language

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(file=f, **kwargs)

    segments = []
    all_words: list[WordSegment] = []

    for seg in getattr(response, "segments", []) or []:
        if isinstance(seg, dict):
            text, start, end = seg.get("text", "").strip(), seg.get("start", 0.0), seg.get("end", 0.0)
        else:
            text, start, end = seg.text.strip(), seg.start, seg.end
        segments.append(TranscriptSegment(text=text, start=float(start), end=float(end)))

    # Extract word-level timestamps if available
    for w in getattr(response, "words", []) or []:
        if isinstance(w, dict):
            all_words.append(WordSegment(
                word=w.get("word", "").strip(),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
            ))
        else:
            all_words.append(WordSegment(
                word=getattr(w, "word", "").strip(),
                start=float(getattr(w, "start", 0.0)),
                end=float(getattr(w, "end", 0.0)),
            ))

    return Transcript(
        text=response.text.strip(),
        segments=segments,
        words=all_words if all_words else None,
    )


def _transcribe_openai_large_file(
    audio_path: Path,
    *,
    model: str,
    client: OpenAI,
    language: str | None = None,
    chunk_duration_sec: int = 600,
) -> Transcript:
    """Split a large audio file into chunks and transcribe each part."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=print_section=0", str(audio_path)],
        capture_output=True, text=True, check=True,
    )
    total_duration = float(result.stdout.strip())

    all_segments: list[TranscriptSegment] = []
    all_words: list[WordSegment] = []
    all_text_parts: list[str] = []
    offset = 0.0

    while offset < total_duration:
        chunk_path = Path(tempfile.mktemp(suffix=".wav"))
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(offset), "-i", str(audio_path),
                 "-t", str(chunk_duration_sec), "-c", "copy", str(chunk_path)],
                capture_output=True, text=True, check=True,
            )
            transcript = _transcribe_openai_file(chunk_path, model=model, client=client, language=language)
            for seg in transcript.segments:
                seg.start += offset
                seg.end += offset
                all_segments.append(seg)
            if transcript.words:
                for w in transcript.words:
                    all_words.append(WordSegment(
                        word=w.word,
                        start=w.start + offset,
                        end=w.end + offset,
                    ))
            all_text_parts.append(transcript.text)
        finally:
            chunk_path.unlink(missing_ok=True)
        offset += chunk_duration_sec

    return Transcript(
        text=" ".join(all_text_parts),
        segments=all_segments,
        words=all_words if all_words else None,
    )


def get_transcript_at_time(transcript: Transcript, time: float, window: float = 5.0) -> str:
    """Get transcript text around a specific timestamp."""
    parts = []
    for seg in transcript.segments:
        if seg.start <= time + window and seg.end >= time - window:
            parts.append(seg.text)
    return " ".join(parts)


def is_mid_sentence(transcript: Transcript, time: float, tolerance: float = 1.0) -> bool:
    """Check if a timestamp falls in the middle of a sentence.

    With word-level timestamps, checks if the nearest word ends with
    sentence-ending punctuation. Falls back to segment-level check.
    """
    if transcript.words:
        # Find the word closest to the target time
        best_word = None
        best_dist = float("inf")
        for w in transcript.words:
            mid = (w.start + w.end) / 2
            dist = abs(mid - time)
            if dist < best_dist:
                best_dist = dist
                best_word = w
        if best_word and best_dist <= tolerance:
            return not best_word.word.endswith((".", "!", "?", ","))
        return False

    # Fallback: segment-level
    for seg in transcript.segments:
        if seg.start + tolerance < time < seg.end - tolerance:
            text = seg.text.strip()
            if text and text[-1] not in ".!?":
                return True
    return False


def find_sentence_boundary(
    transcript: Transcript,
    time: float,
    search_window: float = 8.0,
) -> float | None:
    """Find the nearest sentence boundary to a given timestamp.

    With word-level timestamps, finds the end of the nearest word ending
    with sentence punctuation -- much more precise than segment boundaries.
    Falls back to segment-level boundaries if words are not available.
    """
    # ── Word-level boundary detection ─────────────────────────────────────
    if transcript.words:
        word_ends: list[float] = []
        word_starts: list[float] = []

        for w in transcript.words:
            # Word ending with sentence punctuation = precise cut point
            if abs(w.end - time) <= search_window and w.word[-1:] in ".!?,":
                word_ends.append(w.end)
            # Word start after time = start of new utterance
            if w.start > time and abs(w.start - time) <= search_window:
                word_starts.append(w.start)

        if word_ends:
            return min(word_ends, key=lambda t: abs(t - time))
        if word_starts:
            return min(word_starts, key=lambda t: abs(t - time))

    # ── Segment-level fallback ────────────────────────────────────────────
    sentence_ends: list[float] = []
    segment_starts: list[float] = []

    for seg in transcript.segments:
        if abs(seg.end - time) <= search_window:
            text = seg.text.strip()
            if text and text[-1] in ".!?,":
                sentence_ends.append(seg.end)
        if abs(seg.start - time) <= search_window:
            segment_starts.append(seg.start)

    if sentence_ends:
        return min(sentence_ends, key=lambda t: abs(t - time))
    if segment_starts:
        return min(segment_starts, key=lambda t: abs(t - time))

    return None
