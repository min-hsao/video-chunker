"""Core splitting logic: combines silence detection, transcript validation, and cue detection."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

from .analyzer import ChunkInfo
from .transcriber import Transcript, find_sentence_boundary, get_transcript_at_time
from .utils import SilenceInterval, sanitize_filename, snap_to_keyframe, split_video_segment

logger = logging.getLogger(__name__)

# Minimum chunk duration — shorter chunks are noise, not takes
MIN_CHUNK_DURATION = 15.0

# Retake detection: similarity threshold (0-1) to consider a phrase a retake
RETAKE_SIMILARITY_THRESHOLD = 0.6

# Retake detection: how many words at the start of a segment to compare
RETAKE_COMPARE_WORDS = 8


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
            logger.debug("Cue keyword in segment at %.1f: %s", seg.start, seg.text.strip())
            split_points.append(seg.start)

    return split_points


def _first_words(text: str, n: int) -> str:
    """Return the first n words of text, lowercased and stripped of punctuation."""
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return " ".join(words[:n])


def find_retake_split_points(transcript: Transcript) -> list[float]:
    """Detect retakes by finding segments that repeat the opening phrase of a prior segment.

    When a speaker restarts a sentence or section without saying a cue word,
    this finds the restart point so each attempt becomes its own chunk.

    Returns timestamps where retakes begin.
    """
    split_points: list[float] = []
    segments = transcript.segments
    if len(segments) < 2:
        return split_points

    for i in range(1, len(segments)):
        current = segments[i]
        current_start = _first_words(current.text, RETAKE_COMPARE_WORDS)
        if not current_start:
            continue

        # Look back through segments within the past 60 seconds
        for j in range(i - 1, -1, -1):
            prev = segments[j]
            if current.start - prev.start > 60.0:
                break
            prev_start = _first_words(prev.text, RETAKE_COMPARE_WORDS)
            if not prev_start:
                continue

            similarity = SequenceMatcher(None, current_start, prev_start).ratio()
            if similarity >= RETAKE_SIMILARITY_THRESHOLD:
                logger.info(
                    "Retake detected at %.1fs (%.0f%% similar to %.1fs): %r → %r",
                    current.start, similarity * 100, prev.start,
                    prev_start, current_start,
                )
                split_points.append(current.start)
                break

    return split_points


def _safe_split_point(
    candidate: float,
    transcript: Transcript,
    silence_intervals: list[SilenceInterval],
    search_window: float = 8.0,
) -> float:
    """Move a candidate split point to the nearest silence boundary.

    Priority:
    1. If candidate is inside a silence — use end of that silence.
    2. Find nearest silence within search_window — use its end.
    3. Nudge to transcript sentence boundary if possible.
    4. Fallback to nearest segment start.
    5. Last resort: original candidate.
    """
    # 1. Candidate inside a silence interval
    for interval in silence_intervals:
        if interval.start <= candidate <= interval.end:
            point = interval.end
            boundary = find_sentence_boundary(transcript, point, search_window)
            return boundary if boundary is not None else point

    # 2. Nearest silence interval within window
    best_interval = None
    best_dist = float("inf")
    for interval in silence_intervals:
        dist = min(abs(interval.start - candidate), abs(interval.end - candidate))
        if dist < best_dist and dist <= search_window:
            best_dist = dist
            best_interval = interval

    if best_interval is not None:
        point = best_interval.end
        boundary = find_sentence_boundary(transcript, point, search_window)
        return boundary if boundary is not None else point

    # 3/4. Nearest transcript segment start
    best_seg_start = None
    best_seg_dist = float("inf")
    for seg in transcript.segments:
        dist = abs(seg.start - candidate)
        if dist < best_seg_dist and dist <= search_window:
            best_seg_dist = dist
            best_seg_start = seg.start

    if best_seg_start is not None:
        return best_seg_start

    logger.debug("No safe split point found near %.1f — using original", candidate)
    return candidate


def compute_split_points(
    silence_intervals: list[SilenceInterval],
    transcript: Transcript,
    video_duration: float,
    cue_keywords: list[str] | None = None,
    keyframes: list[float] | None = None,
) -> list[tuple[float, bool, bool]]:
    """Compute final split points.

    Returns list of (timestamp, cue_triggered, is_retake) tuples, sorted by timestamp.

    Sources:
    1. Silence intervals — primary split signal
    2. Verbal cue keywords — force split regardless of silence
    3. Retake detection — repeated phrases indicate a restart
    All splits are anchored to silence boundaries to avoid mid-sentence cuts.
    """
    # (timestamp, is_cue, is_retake)
    raw_points: list[tuple[float, bool, bool]] = []

    # ── 1. Silence-based splits ───────────────────────────────────────────────
    for interval in silence_intervals:
        point = interval.end
        boundary = find_sentence_boundary(transcript, point, search_window=8.0)
        if boundary is not None:
            point = boundary
        raw_points.append((point, False, False))

    # ── 2. Verbal cue splits ──────────────────────────────────────────────────
    if cue_keywords:
        for cp in find_cue_split_points(transcript, cue_keywords):
            safe = _safe_split_point(cp, transcript, silence_intervals)
            raw_points.append((safe, True, False))

    # ── 3. Retake splits ──────────────────────────────────────────────────────
    for rp in find_retake_split_points(transcript):
        safe = _safe_split_point(rp, transcript, silence_intervals)
        raw_points.append((safe, True, True))

    # ── Sort and deduplicate ──────────────────────────────────────────────────
    raw_points.sort(key=lambda x: x[0])

    if not raw_points:
        return []

    deduped: list[tuple[float, bool, bool]] = [raw_points[0]]
    for point, is_cue, is_retake in raw_points[1:]:
        prev_point, prev_cue, prev_retake = deduped[-1]
        if point - prev_point < 1.0:
            # Prefer cue/retake over silence when merging nearby points
            if (is_cue or is_retake) and not (prev_cue or prev_retake):
                deduped[-1] = (point, is_cue, is_retake)
            continue
        deduped.append((point, is_cue, is_retake))

    # ── Snap to keyframes ─────────────────────────────────────────────────────
    if keyframes:
        deduped = [
            (snap_to_keyframe(point, keyframes), is_cue, is_retake)
            for point, is_cue, is_retake in deduped
        ]

    # ── Filter micro-chunks ───────────────────────────────────────────────────
    # Drop splits that produce chunks under MIN_CHUNK_DURATION.
    # Cue and retake splits are kept even if short.
    filtered: list[tuple[float, bool, bool]] = []
    boundaries = [0.0] + [p for p, _, _ in deduped] + [video_duration]
    for i, (point, is_cue, is_retake) in enumerate(deduped):
        chunk_before = point - boundaries[i]
        chunk_after = boundaries[i + 2] - point
        if is_cue or is_retake or (chunk_before >= MIN_CHUNK_DURATION and chunk_after >= MIN_CHUNK_DURATION):
            filtered.append((point, is_cue, is_retake))

    return filtered


def build_chunks(
    split_points: list[tuple[float, bool, bool]],
    transcript: Transcript,
    video_duration: float,
) -> list[ChunkInfo]:
    """Build ChunkInfo objects from split points."""
    chunks: list[ChunkInfo] = []
    boundaries = [0.0] + [p for p, _, _ in split_points] + [video_duration]
    flags = [(False, False)] + [(cue, retake) for _, cue, retake in split_points]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        duration = end - start
        chunk_text = get_transcript_at_time(
            transcript, (start + end) / 2, window=(end - start) / 2 + 1
        )
        is_cue, is_retake = flags[i] if i < len(flags) else (False, False)
        chunks.append(ChunkInfo(
            index=i,
            start=start,
            end=end,
            duration=duration,
            transcript=chunk_text,
            cue_triggered=is_cue,
            retake=is_retake,
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
        split_video_segment(video_path, output_path, start=chunk.start, end=chunk.end)

    return chunks
