"""Audio transcription using local Whisper or OpenAI Whisper API."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .utils import extract_audio

logger = logging.getLogger(__name__)

# Whisper API file size limit (25 MB)
MAX_FILE_SIZE = 25 * 1024 * 1024

# Local Whisper model sizes
WHISPER_MODELS = {
    "tiny": 39e6,  # 39M params
    "base": 74e6,  # 74M params
    "small": 244e6,  # 244M params
    "medium": 769e6,  # 769M params
    "large-v3": 1550e6,  # 1.5B params
}


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


def transcribe_audio(
    video_path: Path,
    *,
    model: str = "base",
    mode: str = "local",
    client: OpenAI | None = None,
) -> Transcript:
    """Transcribe audio from a video file.

    Args:
        video_path: Path to the video file
        model: Whisper model name (e.g., "base", "small", "whisper-1" for API)
        mode: "local" or "openai"
        client: OpenAI client (required for openai mode)

    Extracts audio and returns a full transcript with timestamps.
    """
    # Extract audio to a temporary WAV file
    logger.info("Extracting audio from %s", video_path.name)
    audio_path = extract_audio(video_path)

    try:
        if mode == "local":
            return _transcribe_local(audio_path, model=model)
        else:  # mode == "openai"
            if client is None:
                client = OpenAI()
            return _transcribe_openai(audio_path, model=model, client=client)
    finally:
        # Clean up temp file
        audio_path.unlink(missing_ok=True)


def _transcribe_local(
    audio_path: Path,
    *,
    model: str = "base",
) -> Transcript:
    """Transcribe using local Whisper library."""
    import whisper

    logger.info("Transcribing %s with local Whisper model %s", audio_path.name, model)

    # Load the model (cached locally)
    whisper_model = whisper.load_model(model)

    # Transcribe with word-level timestamps
    result = whisper_model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True,
    )

    # Convert to our format
    segments = []
    for seg in result.get("segments", []):
        segments.append(TranscriptSegment(
            text=seg.get("text", "").strip(),
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
        ))

    return Transcript(
        text=result.get("text", "").strip(),
        segments=segments,
    )


def _transcribe_openai(
    audio_path: Path,
    *,
    model: str,
    client: OpenAI,
) -> Transcript:
    """Transcribe using OpenAI Whisper API."""
    logger.info("Transcribing %s with OpenAI Whisper API model %s", audio_path.name, model)

    file_size = audio_path.stat().st_size

    if file_size <= MAX_FILE_SIZE:
        return _transcribe_openai_file(audio_path, model=model, client=client)

    # For large files, split into chunks
    logger.info("Audio file is %.1f MB, splitting into chunks", file_size / 1024 / 1024)
    return _transcribe_openai_large_file(audio_path, model=model, client=client)


def _transcribe_openai_file(
    audio_path: Path,
    *,
    model: str,
    client: OpenAI,
) -> Transcript:
    """Transcribe a single audio file via API."""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    for seg in getattr(response, "segments", []) or []:
        segments.append(TranscriptSegment(
            text=seg.get("text", "").strip() if isinstance(seg, dict) else seg.text.strip(),
            start=seg.get("start", 0.0) if isinstance(seg, dict) else seg.start,
            end=seg.get("end", 0.0) if isinstance(seg, dict) else seg.end,
        ))

    return Transcript(
        text=response.text.strip(),
        segments=segments,
    )


def _transcribe_openai_large_file(
    audio_path: Path,
    *,
    model: str,
    client: OpenAI,
    chunk_duration_sec: int = 600,
) -> Transcript:
    """Split a large audio file and transcribe in parts."""
    import subprocess

    # Get audio duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=print_section=0", str(audio_path)],
        capture_output=True, text=True, check=True,
    )
    total_duration = float(result.stdout.strip())

    all_segments: list[TranscriptSegment] = []
    all_text_parts: list[str] = []
    offset = 0.0

    while offset < total_duration:
        chunk_path = Path(tempfile.mktemp(suffix=".wav"))
        try:
            # Extract chunk
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(offset), "-i", str(audio_path),
                 "-t", str(chunk_duration_sec), "-c", "copy", str(chunk_path)],
                capture_output=True, text=True, check=True,
            )

            transcript = _transcribe_openai_file(chunk_path, model=model, client=client)

            # Adjust timestamps by offset
            for seg in transcript.segments:
                seg.start += offset
                seg.end += offset
                all_segments.append(seg)

            all_text_parts.append(transcript.text)
        finally:
            chunk_path.unlink(missing_ok=True)

        offset += chunk_duration_sec

    return Transcript(
        text=" ".join(all_text_parts),
        segments=all_segments,
    )


def get_transcript_at_time(transcript: Transcript, time: float, window: float = 5.0) -> str:
    """Get transcript text around a specific timestamp."""
    parts = []
    for seg in transcript.segments:
        if seg.start <= time + window and seg.end >= time - window:
            parts.append(seg.text)
    return " ".join(parts)


def is_mid_sentence(transcript: Transcript, time: float, tolerance: float = 1.0) -> bool:
    """Check if a timestamp falls in the middle of a sentence."""
    for seg in transcript.segments:
        # Check if the time falls within a segment (not at the boundary)
        if seg.start + tolerance < time < seg.end - tolerance:
            # Check if the segment text doesn't end with sentence-ending punctuation
            text = seg.text.strip()
            if text and not text[-1] in ".!?":
                return True
    return False


def find_sentence_boundary(
    transcript: Transcript,
    time: float,
    search_window: float = 3.0,
) -> float | None:
    """Find the nearest sentence boundary to a given timestamp."""
    candidates: list[float] = []

    for seg in transcript.segments:
        # Check segment boundaries within the search window
        if abs(seg.end - time) <= search_window:
            text = seg.text.strip()
            if text and text[-1] in ".!?":
                candidates.append(seg.end)

        if abs(seg.start - time) <= search_window:
            candidates.append(seg.start)

    if not candidates:
        return None

    # Return the candidate closest to the original time
    return min(candidates, key=lambda t: abs(t - time))
