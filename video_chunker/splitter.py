"""Core splitting logic: combines silence detection, transcript validation, and cue detection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
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


# ── Clean mode: remove filler words, silences, retakes and stitch ───────────

# Common filler words/phrases to detect
FILLER_PHRASES: list[str] = [
    "um", "uh", "ah", "er", "hmm", "huh",
    "you know", "like", "I mean", "so yeah",
    "basically", "literally", "actually",
    "kind of", "sort of",
]

# Minimum gap to keep between segments (avoid cutting into speech)
MIN_KEEP_DURATION = 0.3


def _is_filler_segment(text: str) -> bool:
    """Check if a transcript segment is purely a filler word/phrase."""
    cleaned = text.strip().lower().rstrip(".,!?;:")
    return cleaned in FILLER_PHRASES


def _is_retake_segment(
    seg_index: int,
    transcript: Transcript,
) -> bool:
    """Check if a segment is a retake of a previous segment."""
    segments = transcript.segments
    if seg_index >= len(segments):
        return False

    current = segments[seg_index]
    current_start = _first_words(current.text, RETAKE_COMPARE_WORDS)
    if not current_start:
        return False

    for j in range(seg_index - 1, -1, -1):
        prev = segments[j]
        if current.start - prev.start > 60.0:
            break
        prev_start = _first_words(prev.text, RETAKE_COMPARE_WORDS)
        if not prev_start:
            continue
        similarity = SequenceMatcher(None, current_start, prev_start).ratio()
        if similarity >= RETAKE_SIMILARITY_THRESHOLD:
            return True

    return False


@dataclass
class CleanSegment:
    """A 'good' segment to keep in the cleaned output."""
    start: float
    end: float
    reason: str  # "kept" | "speech"


def compute_clean_segments(
    transcript: Transcript,
    silence_intervals: list[SilenceInterval],
    video_duration: float,
    *,
    min_silence_to_cut: float = 1.0,
    remove_retakes: bool = True,
) -> list[CleanSegment]:
    """Identify the 'good' segments to keep after removing filler, silence, and retakes.

    Returns a list of CleanSegment representing the parts to keep.
    The gaps between these segments are what gets cut.
    """
    # Build a set of time ranges to cut
    cut_ranges: list[tuple[float, float, str]] = []  # (start, end, reason)

    # ── 1. Cut silence intervals ──────────────────────────────────────────
    for si in silence_intervals:
        if si.duration >= min_silence_to_cut:
            cut_ranges.append((si.start, si.end, "silence"))

    # ── 2. Cut filler word segments ───────────────────────────────────────
    for seg in transcript.segments:
        if _is_filler_segment(seg.text):
            cut_ranges.append((seg.start, seg.end, "filler"))

    # ── 3. Cut retakes ────────────────────────────────────────────────────
    if remove_retakes:
        for i in range(len(transcript.segments)):
            if _is_retake_segment(i, transcript):
                seg = transcript.segments[i]
                cut_ranges.append((seg.start, seg.end, "retake"))

    # ── Merge overlapping cut ranges ──────────────────────────────────────
    if not cut_ranges:
        # Nothing to cut — keep the whole video
        return [CleanSegment(start=0.0, end=video_duration, reason="kept")]

    cut_ranges.sort(key=lambda x: x[0])
    merged: list[tuple[float, float, str]] = [cut_ranges[0]]
    for start, end, reason in cut_ranges[1:]:
        prev_start, prev_end, prev_reason = merged[-1]
        if start <= prev_end:
            # Overlapping or adjacent — merge
            merged[-1] = (prev_start, max(prev_end, end), prev_reason)
        else:
            merged.append((start, end, reason))

    # ── Build keep segments (inverse of cut ranges) ───────────────────────
    keep: list[CleanSegment] = []
    cursor = 0.0

    for cut_start, cut_end, reason in merged:
        if cut_start - cursor >= MIN_KEEP_DURATION:
            keep.append(CleanSegment(start=cursor, end=cut_start, reason="speech"))
        cursor = cut_end

    # Trailing segment after last cut
    if video_duration - cursor >= MIN_KEEP_DURATION:
        keep.append(CleanSegment(start=cursor, end=video_duration, reason="speech"))

    if not keep:
        # Edge case: entire video was cut — keep it all
        keep.append(CleanSegment(start=0.0, end=video_duration, reason="kept"))

    return keep


def clean_video(
    video_path: Path,
    transcript: Transcript,
    silence_intervals: list[SilenceInterval],
    output_path: Path,
    *,
    min_silence_to_cut: float = 1.0,
    remove_retakes: bool = True,
) -> dict:
    """Remove filler words, silences, and retakes from a video.

    Outputs a single cleaned file with the bad parts cut out.

    Returns a summary dict with stats.
    """
    info_cmd = __import__("subprocess").run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=print_section=0", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    video_duration = float(info_cmd.stdout.strip())

    keep_segments = compute_clean_segments(
        transcript,
        silence_intervals,
        video_duration,
        min_silence_to_cut=min_silence_to_cut,
        remove_retakes=remove_retakes,
    )

    original_duration = video_duration
    kept_duration = sum(s.end - s.start for s in keep_segments)
    cut_duration = original_duration - kept_duration

    logger.info(
        "Clean mode: keeping %.1fs of %.1fs (cut %.1fs / %.0f%%)",
        kept_duration, original_duration, cut_duration,
        (cut_duration / original_duration * 100) if original_duration > 0 else 0,
    )

    if len(keep_segments) == 1 and keep_segments[0].start == 0.0 and abs(keep_segments[0].end - video_duration) < 0.5:
        # Nothing to cut — just copy
        import shutil
        shutil.copy2(video_path, output_path)
        logger.info("No cuts needed — copied original file")
        return {
            "original_duration": original_duration,
            "clean_duration": kept_duration,
            "cut_duration": 0.0,
            "segments_kept": 1,
            "cuts_made": 0,
        }

    # Create individual clips then concat
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="vclean_"))
    clip_list = tmpdir / "concat.txt"

    try:
        # Write each keep segment as a clip
        clip_paths: list[Path] = []
        for i, seg in enumerate(keep_segments):
            clip_path = tmpdir / f"keep_{i:04d}.mp4"
            split_video_segment(video_path, clip_path, start=seg.start, end=seg.end)
            clip_paths.append(clip_path)

        # Write concat list
        with open(clip_list, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        # Concatenate with stream copy
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(clip_list), "-c", "copy",
             "-movflags", "+faststart", str(output_path)],
            capture_output=True, text=True, check=True,
        )

    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "original_duration": original_duration,
        "clean_duration": kept_duration,
        "cut_duration": cut_duration,
        "segments_kept": len(keep_segments),
        "cuts_made": len(keep_segments) - 1,
    }


def compute_clean_segments_llm(
    transcript: Transcript,
    silence_intervals: list[SilenceInterval],
    video_duration: float,
    *,
    llm_retakes: list | None = None,
    llm_fillers: list | None = None,
    llm_content_cuts: list | None = None,
    min_silence_to_cut: float = 1.0,
    remove_retakes: bool = True,
) -> list[CleanSegment]:
    """Enhanced clean mode using LLM analysis results.

    Like compute_clean_segments but integrates:
    - LLM retake detection (replaces string similarity)
    - LLM filler detection (replaces hardcoded word list)
    - LLM content cut suggestions (tangents, redundancy, etc.)
    """
    cut_ranges: list[tuple[float, float, str]] = []  # (start, end, reason)

    # ── 1. Cut silence intervals ──────────────────────────────────────────
    for si in silence_intervals:
        if si.duration >= min_silence_to_cut:
            cut_ranges.append((si.start, si.end, "silence"))

    # ── 2. LLM filler detection (preferred) or fallback ──────────────────
    if llm_fillers:
        from .analyzer import FillerDetection
        for fd in llm_fillers:
            if isinstance(fd, FillerDetection) and fd.is_filler:
                cut_ranges.append((fd.start, fd.end, f"filler: {fd.reason}"))
    else:
        # Fallback to hardcoded filler detection
        for seg in transcript.segments:
            if _is_filler_segment(seg.text):
                cut_ranges.append((seg.start, seg.end, "filler"))

    # ── 3. LLM retake detection (preferred) or fallback ──────────────────
    if remove_retakes:
        if llm_retakes:
            from .analyzer import RetakeDetection
            for rd in llm_retakes:
                if isinstance(rd, RetakeDetection) and rd.is_retake:
                    cut_ranges.append((rd.timestamp, rd.timestamp + 0.1, f"retake: {rd.reason}"))
        else:
            # Fallback to string similarity retake detection
            for i in range(len(transcript.segments)):
                if _is_retake_segment(i, transcript):
                    seg = transcript.segments[i]
                    cut_ranges.append((seg.start, seg.end, "retake"))

    # ── 4. LLM content cut suggestions ────────────────────────────────────
    if llm_content_cuts:
        from .analyzer import ContentCutSuggestion
        for cc in llm_content_cuts:
            if isinstance(cc, ContentCutSuggestion):
                cut_ranges.append((cc.start, cc.end, f"content: {cc.reason}"))

    # ── Merge overlapping cut ranges ──────────────────────────────────────
    if not cut_ranges:
        return [CleanSegment(start=0.0, end=video_duration, reason="kept")]

    cut_ranges.sort(key=lambda x: x[0])
    merged: list[tuple[float, float, str]] = [cut_ranges[0]]
    for start, end, reason in cut_ranges[1:]:
        prev_start, prev_end, prev_reason = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end), prev_reason)
        else:
            merged.append((start, end, reason))

    # ── Build keep segments (inverse of cut ranges) ───────────────────────
    keep: list[CleanSegment] = []
    cursor = 0.0

    for cut_start, cut_end, reason in merged:
        if cut_start - cursor >= MIN_KEEP_DURATION:
            keep.append(CleanSegment(start=cursor, end=cut_start, reason="speech"))
        cursor = cut_end

    # Trailing segment after last cut
    if video_duration - cursor >= MIN_KEEP_DURATION:
        keep.append(CleanSegment(start=cursor, end=video_duration, reason="speech"))

    if not keep:
        keep.append(CleanSegment(start=0.0, end=video_duration, reason="kept"))

    return keep
