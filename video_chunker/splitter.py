"""Core splitting logic: combines silence detection, transcript validation, and cue detection."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .analyzer import ChunkInfo
from .transcriber import Transcript, find_sentence_boundary, get_transcript_at_time, is_mid_sentence
from .utils import SilenceInterval, get_keyframes, sanitize_filename, snap_to_keyframe, split_video_segment

logger = logging.getLogger(__name__)


def find_cue_split_points(
    transcript: Transcript,
    cue_keywords: list[str],
) -> list[float]:
    """Find timestamps where verbal cue keywords appear in the transcript."""
    split_points: list[float] = []

    if not cue_keywords:
        return split_points

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in cue_keywords) + r")\b",
        re.IGNORECASE,
    )

    for seg in transcript.segments:
        if pattern.search(seg.text):
            logger.debug("Found cue keyword in segment at %.1f: %s", seg.start, seg.text.strip())
            # Split point is at the start of the segment containing the cue
            split_points.append(seg.start)

    return split_points


def compute_split_points(
    silence_intervals: list[SilenceInterval],
    transcript: Transcript,
    video_duration: float,
    cue_keywords: list[str] | None = None,
    keyframes: list[float] | None = None,
) -> list[tuple[float, bool]]:
    """Compute final split points from silence intervals, transcript, and cues.

    Returns list of (timestamp, cue_triggered) tuples, sorted by timestamp.
    """
    raw_points: list[tuple[float, bool]] = []

    # Add silence-based split points (using midpoint of silence)
    for interval in silence_intervals:
        point = interval.midpoint
        # Validate against transcript - don't split mid-sentence
        if is_mid_sentence(transcript, point):
            boundary = find_sentence_boundary(transcript, point)
            if boundary is not None:
                logger.debug(
                    "Adjusted split point from %.1f to %.1f (sentence boundary)",
                    point, boundary,
                )
                point = boundary
        raw_points.append((point, False))

    # Add cue-based split points (these force a split regardless)
    if cue_keywords:
        cue_points = find_cue_split_points(transcript, cue_keywords)
        for cp in cue_points:
            raw_points.append((cp, True))

    # Sort by timestamp
    raw_points.sort(key=lambda x: x[0])

    # Deduplicate points that are too close together (within 1 second)
    if not raw_points:
        return []

    deduped: list[tuple[float, bool]] = [raw_points[0]]
    for point, is_cue in raw_points[1:]:
        prev_point, prev_cue = deduped[-1]
        if point - prev_point < 1.0:
            # Keep the cue-triggered one if either is a cue
            if is_cue and not prev_cue:
                deduped[-1] = (point, True)
            continue
        deduped.append((point, is_cue))

    # Snap to keyframes if available
    if keyframes:
        deduped = [
            (snap_to_keyframe(point, keyframes), is_cue)
            for point, is_cue in deduped
        ]

    # Filter out points that would create chunks shorter than minimum duration.
    # Short micro-chunks (< 10s) are usually gaps, hesitations, or false splits —
    # not meaningful takes. Cue-triggered splits are exempt.
    MIN_CHUNK_DURATION = 10.0
    filtered: list[tuple[float, bool]] = []
    boundaries = [0.0] + [p for p, _ in deduped] + [video_duration]
    for i, (point, is_cue) in enumerate(deduped):
        chunk_before = point - boundaries[i]
        chunk_after = boundaries[i + 2] - point
        if is_cue or (chunk_before >= MIN_CHUNK_DURATION and chunk_after >= MIN_CHUNK_DURATION):
            filtered.append((point, is_cue))
    deduped = filtered

    return deduped


def build_chunks(
    split_points: list[tuple[float, bool]],
    transcript: Transcript,
    video_duration: float,
) -> list[ChunkInfo]:
    """Build chunk info objects from split points."""
    chunks: list[ChunkInfo] = []

    # Build time ranges: [0, split1], [split1, split2], ..., [splitN, duration]
    boundaries = [0.0] + [p for p, _ in split_points] + [video_duration]
    cue_flags = [False] + [c for _, c in split_points]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        duration = end - start

        # Get transcript for this chunk
        chunk_text = get_transcript_at_time(transcript, (start + end) / 2, window=(end - start) / 2 + 1)

        chunks.append(ChunkInfo(
            index=i,
            start=start,
            end=end,
            duration=duration,
            transcript=chunk_text,
            cue_triggered=cue_flags[i] if i < len(cue_flags) else False,
        ))

    return chunks


def generate_output_filename(chunk: ChunkInfo, extension: str = "mp4") -> str:
    """Generate output filename from chunk analysis."""
    index_str = f"{chunk.index + 1:03d}"

    if chunk.analysis:
        desc = sanitize_filename(chunk.analysis.description)
        status = "complete" if chunk.analysis.is_complete else "incomplete"
    else:
        desc = "untitled"
        status = "unknown"

    return f"{index_str}_{desc}_{status}.{extension}"


def split_video(
    video_path: Path,
    chunks: list[ChunkInfo],
    output_dir: Path,
    extension: str = "mp4",
) -> list[ChunkInfo]:
    """Split the video file into chunks and save to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        filename = generate_output_filename(chunk, extension)
        output_path = output_dir / filename
        chunk.output_path = str(output_path)

        logger.info(
            "Splitting chunk %d: %.1fs - %.1fs -> %s",
            chunk.index + 1, chunk.start, chunk.end, filename,
        )

        split_video_segment(
            video_path,
            output_path,
            start=chunk.start,
            end=chunk.end,
        )

    return chunks
