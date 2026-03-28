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
