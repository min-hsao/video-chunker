"""FFmpeg/FFprobe utilities and silence detection."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    codec: str
    width: int
    height: int
    duration: float
    fps: float
    container: str
    audio_codec: str | None


@dataclass
class SilenceInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


def _run(cmd: list[str], *, capture_stderr: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and return the result."""
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result
    except FileNotFoundError:
        raise RuntimeError(
            f"Command not found: {cmd[0]}. Please install ffmpeg/ffprobe."
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        raise RuntimeError(f"{cmd[0]} failed (exit {e.returncode}): {stderr[:500]}")


def get_video_info(video_path: Path) -> VideoInfo:
    """Probe a video file and return its metadata."""
    result = _run([
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ])
    data = json.loads(result.stdout)

    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0))

    # Parse fps from r_frame_rate (e.g. "30000/1001")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    return VideoInfo(
        codec=video_stream.get("codec_name", "unknown"),
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        duration=duration,
        fps=fps,
        container=Path(video_path).suffix.lstrip(".").lower(),
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
    )


def extract_audio(video_path: Path, output_path: Path | None = None) -> Path:
    """Extract audio from a video file as WAV (16kHz mono for Whisper)."""
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".wav"))

    _run([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ])
    return output_path


def detect_silence(
    video_path: Path,
    silence_duration: float = 2.0,
    silence_threshold: float = -35,
) -> list[SilenceInterval]:
    """Detect silence intervals in a video/audio file using ffmpeg silencedetect."""
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-af", f"silencedetect=noise={silence_threshold}dB:d={silence_duration}",
        "-f", "null",
        "-",
    ]

    logger.debug("Running silence detection: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    # silencedetect outputs to stderr
    output = result.stderr

    # Parse silence_start and silence_end pairs
    starts: list[float] = []
    ends: list[float] = []

    for line in output.splitlines():
        start_match = re.search(r"silence_start:\s*([\d.]+)", line)
        if start_match:
            starts.append(float(start_match.group(1)))

        end_match = re.search(r"silence_end:\s*([\d.]+)", line)
        if end_match:
            ends.append(float(end_match.group(1)))

    intervals = []
    for i, start in enumerate(starts):
        end = ends[i] if i < len(ends) else start + silence_duration
        intervals.append(SilenceInterval(start=start, end=end))

    logger.info("Detected %d silence intervals", len(intervals))
    return intervals


def get_keyframes(video_path: Path) -> list[float]:
    """Get a list of keyframe timestamps from a video file."""
    result = _run([
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-of", "csv=print_section=0",
        str(video_path),
    ])

    keyframes = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                keyframes.append(float(parts[0]))
            except (ValueError, IndexError):
                continue

    keyframes.sort()
    logger.info("Found %d keyframes", len(keyframes))
    return keyframes


def snap_to_keyframe(timestamp: float, keyframes: list[float]) -> float:
    """Snap a timestamp to the nearest keyframe (preferring the one before)."""
    if not keyframes:
        return timestamp

    # Find the nearest keyframe at or before the timestamp
    best = keyframes[0]
    for kf in keyframes:
        if kf <= timestamp:
            best = kf
        else:
            # Check if this keyframe is closer than the previous one
            if abs(kf - timestamp) < abs(best - timestamp):
                best = kf
            break

    return best


def split_video_segment(
    video_path: Path,
    output_path: Path,
    start: float,
    end: float | None = None,
) -> None:
    """Split a video segment using lossless stream copy."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
    ]

    if end is not None:
        duration = end - start
        cmd.extend(["-t", f"{duration:.3f}"])

    cmd.extend([
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        str(output_path),
    ])

    _run(cmd)


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Convert text to a safe filename component."""
    # Lowercase and replace spaces/special chars with underscores
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    text = text.strip("_")
    return text[:max_length]


# ── Quality control ───────────────────────────────────────────────────────────


@dataclass
class QCResult:
    """Quality check result for a single output file."""
    path: str
    exists: bool
    duration: float
    expected_duration: float
    duration_ok: bool
    width: int
    height: int
    has_video: bool
    has_audio: bool
    file_size_mb: float
    errors: list[str]


def qc_file(
    file_path: Path,
    expected_duration: float | None = None,
    tolerance: float = 0.5,
) -> QCResult:
    """Run ffprobe QC on an output file.

    Checks: file exists, playable video stream, audio stream present,
    duration matches expected (within tolerance), reasonable file size.
    """
    errors: list[str] = []

    if not file_path.exists():
        return QCResult(
            path=str(file_path),
            exists=False,
            duration=0,
            expected_duration=expected_duration or 0,
            duration_ok=False,
            width=0,
            height=0,
            has_video=False,
            has_audio=False,
            file_size_mb=0,
            errors=["File does not exist"],
        )

    try:
        result = _run([
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ])
        data = json.loads(result.stdout)
    except Exception as e:
        return QCResult(
            path=str(file_path),
            exists=True,
            duration=0,
            expected_duration=expected_duration or 0,
            duration_ok=False,
            width=0,
            height=0,
            has_video=False,
            has_audio=False,
            file_size_mb=file_path.stat().st_size / (1024 * 1024),
            errors=[f"ffprobe failed: {e}"],
        )

    fmt = data.get("format", {})
    actual_duration = float(fmt.get("duration", 0))
    file_size_mb = float(fmt.get("size", 0)) / (1024 * 1024)

    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    has_video = video_stream is not None
    has_audio = audio_stream is not None
    width = int(video_stream.get("width", 0)) if video_stream else 0
    height = int(video_stream.get("height", 0)) if video_stream else 0

    duration_ok = True
    if expected_duration is not None:
        diff = abs(actual_duration - expected_duration)
        if diff > tolerance:
            duration_ok = False
            errors.append(
                f"Duration mismatch: expected {expected_duration:.2f}s, got {actual_duration:.2f}s "
                f"(diff {diff:.2f}s > tolerance {tolerance}s)"
            )

    if not has_video:
        errors.append("No video stream found")
    if not has_audio:
        errors.append("No audio stream found")

    return QCResult(
        path=str(file_path),
        exists=True,
        duration=actual_duration,
        expected_duration=expected_duration or actual_duration,
        duration_ok=duration_ok,
        width=width,
        height=height,
        has_video=has_video,
        has_audio=has_audio,
        file_size_mb=file_size_mb,
        errors=errors,
    )


# ── Contact sheet generation ─────────────────────────────────────────────────


def generate_contact_sheet(
    video_path: Path,
    output_path: Path,
    num_thumbnails: int = 4,
    grid_cols: int = 2,
    thumb_width: int = 320,
) -> Path:
    """Generate a contact sheet (thumbnail grid) from a video file.

    Extracts num_thumbnails frames evenly spaced through the video,
    arranges them in a grid, and saves as a JPEG.

    Returns the path to the generated contact sheet.
    """
    # Get video info for duration
    info = get_video_info(video_path)
    duration = info.duration

    if duration <= 0:
        raise RuntimeError(f"Cannot generate contact sheet: invalid duration {duration}")

    # Calculate evenly spaced timestamps (avoid very first/last frames)
    margin = min(duration * 0.05, 2.0)  # 5% or 2s margin
    usable = duration - 2 * margin
    if usable <= 0:
        timestamps = [duration / 2]
    else:
        step = usable / max(num_thumbnails - 1, 1)
        timestamps = [margin + i * step for i in range(num_thumbnails)]

    # Extract thumbnails to temp dir
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="vc thumbs "))
    thumb_paths: list[Path] = []

    try:
        for i, ts in enumerate(timestamps):
            thumb_path = tmpdir / f"thumb_{i:02d}.jpg"
            _run([
                "ffmpeg",
                "-y",
                "-ss", f"{ts:.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-vf", f"scale={thumb_width}:-1",
                "-q:v", "3",
                str(thumb_path),
            ])
            thumb_paths.append(thumb_path)

        if not thumb_paths:
            raise RuntimeError("No thumbnails extracted")

        # Get dimensions of first thumbnail for grid layout
        thumb_result = _run([
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=print_section=0",
            str(thumb_paths[0]),
        ])
        parts = thumb_result.stdout.strip().split(",")
        tw = int(parts[0]) if len(parts) >= 2 else thumb_width
        th = int(parts[1]) if len(parts) >= 2 else int(thumb_width * 9 / 16)

        grid_rows = (len(thumb_paths) + grid_cols - 1) // grid_cols
        padding = 4
        label_height = 0  # no labels for now

        canvas_w = grid_cols * tw + (grid_cols + 1) * padding
        canvas_h = grid_rows * (th + label_height) + (grid_rows + 1) * padding

        # Build ffmpeg filter to tile thumbnails into grid
        inputs = []
        filter_parts = []
        for i, tp in enumerate(thumb_paths):
            inputs.extend(["-i", str(tp)])
            filter_parts.append(f"[{i}:v]")

        # Build grid using hstack per row, then vstack the rows
        row_filters = []
        row_labels = []
        for row_idx in range(grid_rows):
            cols_in_row = min(grid_cols, len(thumb_paths) - row_idx * grid_cols)
            row_inputs = []
            for col_idx in range(cols_in_row):
                idx = row_idx * grid_cols + col_idx
                row_inputs.append(f"[{idx}:v]")
            row_label = f"row{row_idx}"
            row_filters.append(
                "".join(row_inputs) + f"hstack=inputs={cols_in_row}[{row_label}]"
            )
            row_labels.append(f"[{row_label}]")

        if len(row_labels) == 1:
            # Single row — just scale to canvas
            filter_complex = ";".join(row_filters) + f";{row_labels[0]}scale={canvas_w}:{canvas_h}[out]"
        else:
            # Multiple rows — vstack then pad
            filter_complex = (
                ";".join(row_filters)
                + ";" + "".join(row_labels) + f"vstack=inputs={len(row_labels)}[v]"
                + f";[v]pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2:color=black[out]"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _run([
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ])

        logger.info("Contact sheet saved to %s", output_path)
        return output_path

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
