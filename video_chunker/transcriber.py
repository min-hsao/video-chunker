"""Audio transcription: whisper.cpp (Mac GPU), local Python Whisper, or the OpenAI API.

Engines (``mode``):
- ``cpp``    whisper.cpp's ``whisper-cli`` (Metal on Apple Silicon). Fast, free, offline.
             Optional Silero VAD so long pauses are skipped instead of hallucinated over.
- ``local``  the openai-whisper Python package on the CPU (MPS returns empty transcripts).
             Slow on long recordings; install with ``pip install 'video-chunker[local-whisper]'``.
- ``openai`` the paid OpenAI transcription API.
- ``auto``   ``cpp`` when ``whisper-cli`` is installed, otherwise ``local``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .utils import extract_audio

logger = logging.getLogger(__name__)

# Whisper API file size limit (25 MB)
MAX_FILE_SIZE = 25 * 1024 * 1024

# Default model per engine (the CLI resolves ``--whisper-model`` against these).
DEFAULT_MODELS = {"cpp": "large-v3-turbo", "local": "small", "openai": "whisper-1"}

WHISPER_CPP_BIN = "whisper-cli"
MODEL_DIR = Path(os.environ.get("VIDEO_CHUNKER_MODEL_DIR", Path.home() / ".cache" / "video-chunker" / "models"))
GGML_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
VAD_MODEL_NAME = "ggml-silero-v5.1.2.bin"
VAD_URL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/" + VAD_MODEL_NAME


@dataclass
class WordSegment:
    word: str
    start: float
    end: float
    confidence: float | None = None


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


def resolve_mode(mode: str) -> str:
    """Turn ``auto`` into a concrete engine."""
    mode = mode.lower()
    if mode != "auto":
        return mode
    return "cpp" if shutil.which(WHISPER_CPP_BIN) else "local"


def transcribe_audio(
    video_path: Path,
    *,
    model: str | None = None,
    mode: str = "auto",
    language: str | None = None,
    client: OpenAI | None = None,
    prompt: str | None = None,
    clean_audio: bool = True,
    vad: bool = False,
) -> Transcript:
    """Transcribe audio from a video file.

    Args:
        video_path: Path to the video file
        model: Model name (see DEFAULT_MODELS) or, for cpp, a path to a ggml ``.bin``.
        mode: "auto", "cpp", "local" or "openai"
        language: Optional language code (e.g. "en"). Auto-detect if None.
        client: OpenAI client (openai mode)
        prompt: Names and terms to bias spelling ("Sam, Samtrix, ...").
        clean_audio: Clean the transcription copy (high-pass, denoise, loudness). The
                     source video is never touched.
        vad: cpp only: skip non-speech with Silero VAD. Segment times stay correct but
             whisper.cpp (1.8.x) leaves word times in VAD-compressed time, so word
             timings are dropped and segment-level logic is used instead.
    """
    mode = resolve_mode(mode)
    model = model or DEFAULT_MODELS[mode]
    logger.info("Extracting audio from %s%s", video_path.name, " (cleaned for speech)" if clean_audio else "")
    audio_path = extract_audio(video_path, clean=clean_audio)

    try:
        if mode == "cpp":
            return _transcribe_cpp(audio_path, model=model, language=language, prompt=prompt, vad=vad)
        if mode == "local":
            return _transcribe_local(audio_path, model=model, language=language, prompt=prompt)
        if client is None:
            client = OpenAI()
        return _transcribe_openai(audio_path, model=model, client=client, language=language, prompt=prompt)
    finally:
        audio_path.unlink(missing_ok=True)


# ── whisper.cpp ───────────────────────────────────────────────────────────────


def _download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` atomically (``.part`` then rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s → %s (first use only)", url, dest)
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:  # noqa: S310 (fixed https URL)
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_pct = -10
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct >= last_pct + 10:
                    logger.info("  %d%% of %.0f MB", pct, total / 1e6)
                    last_pct = pct
    part.rename(dest)
    return dest


def resolve_ggml_model(model: str, *, download: bool = True) -> Path:
    """Find (or download once) the ggml model for whisper.cpp.

    ``model`` is a path to a ``.bin`` file or a whisper.cpp model name such as
    ``large-v3-turbo``, ``large-v3-turbo-q5_0``, ``small.en`` or ``base``.
    """
    as_path = Path(model).expanduser()
    if as_path.suffix == ".bin" or as_path.exists():
        if not as_path.exists():
            raise RuntimeError(f"whisper.cpp model not found: {as_path}")
        return as_path
    dest = MODEL_DIR / f"ggml-{model}.bin"
    if dest.exists():
        return dest
    if not download:
        raise RuntimeError(f"whisper.cpp model '{model}' is not downloaded ({dest})")
    return _download(GGML_URL.format(name=model), dest)


def _vad_model() -> Path | None:
    dest = MODEL_DIR / VAD_MODEL_NAME
    if dest.exists():
        return dest
    try:
        return _download(VAD_URL, dest)
    except Exception as e:  # VAD is an improvement, never a blocker
        logger.warning("Could not get the VAD model (%s) — transcribing without VAD", e)
        return None


DTW_PRESETS = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
    "large.v1", "large.v2", "large.v3", "large.v3.turbo",
}


def dtw_preset(model_path: Path) -> str | None:
    """whisper.cpp DTW alignment preset for a ggml file (``ggml-large-v3-turbo-q5_0.bin``
    → ``large.v3.turbo``), or None when the model is not a known one."""
    name = model_path.stem
    if name.startswith("ggml-"):
        name = name[5:]
    name = re.sub(r"-q\d(_\d|_k)?$", "", name)
    preset = name.replace("-", ".")
    return preset if preset in DTW_PRESETS else None


def _estimated_word_length(word: str) -> float:
    letters = sum(ch.isalnum() for ch in word)
    return min(max(0.15, 0.075 * letters), 1.2)


def parse_whisper_cpp_json(data: dict) -> Transcript:
    """Convert whisper-cli ``-ojf`` output into a Transcript with word timings.

    Tokens that start with a space begin a new word; other tokens (sub-words,
    punctuation) attach to the current word. Special tokens (``[_BEG_]``,
    ``[_TT_…]``) are skipped. Punctuation tokens never stretch a word's end time.

    When the output has DTW times (``t_dtw`` ≥ 0, centiseconds), they set each word's
    start; a word ends where the next one starts, capped at a length estimated from its
    letters. Plain token offsets are only a fallback: whisper.cpp often stretches the
    last word of a sentence across the pause that follows it.
    """
    segments: list[TranscriptSegment] = []
    words: list[WordSegment] = []
    dtw_starts: list[float | None] = []

    for seg in data.get("transcription", []):
        off = seg.get("offsets", {})
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append(TranscriptSegment(
            text=text,
            start=off.get("from", 0) / 1000.0,
            end=off.get("to", 0) / 1000.0,
        ))

        cur: dict | None = None
        probs: list[float] = []

        def flush() -> None:
            if cur and cur["word"].strip():
                words.append(WordSegment(
                    word=cur["word"].strip(),
                    start=cur["start"],
                    end=cur["end"],
                    confidence=round(sum(probs) / len(probs), 3) if probs else None,
                ))
                dtw_starts.append(cur.get("dtw"))

        for tok in seg.get("tokens", []):
            t = tok.get("text", "")
            if not t or t.startswith("[_"):
                continue
            t_off = tok.get("offsets", {})
            t0, t1 = t_off.get("from", 0) / 1000.0, t_off.get("to", 0) / 1000.0
            is_punct = not any(ch.isalnum() for ch in t)
            if t.startswith(" ") or cur is None:
                if is_punct and cur is not None:
                    cur["word"] += t.strip()
                    continue
                flush()
                dtw = tok.get("t_dtw", -1)
                cur = {"word": t, "start": t0, "end": t1,
                       "dtw": dtw / 100.0 if isinstance(dtw, (int, float)) and dtw >= 0 else None}
                probs = []
            else:
                cur["word"] += t
                if not is_punct:
                    cur["end"] = t1
            if not is_punct and "p" in tok:
                probs.append(float(tok["p"]))
        flush()

    if words and all(d is not None for d in dtw_starts):
        for i, w in enumerate(words):
            w.start = dtw_starts[i]
        for i, w in enumerate(words):
            nxt = words[i + 1].start if i + 1 < len(words) else None
            est = w.start + _estimated_word_length(w.word)
            w.end = round(min(est, nxt) if nxt is not None and nxt > w.start else est, 3)

    return Transcript(
        text=" ".join(s.text for s in segments),
        segments=segments,
        words=words or None,
    )


def _transcribe_cpp(
    audio_path: Path,
    *,
    model: str,
    language: str | None = None,
    prompt: str | None = None,
    vad: bool = False,
) -> Transcript:
    """Transcribe with whisper.cpp (Metal GPU on Apple Silicon)."""
    binary = shutil.which(WHISPER_CPP_BIN)
    if not binary:
        raise RuntimeError("whisper-cli not found. Install it with: brew install whisper-cpp")
    model_path = resolve_ggml_model(model)
    logger.info("Transcribing with whisper.cpp model %s", model_path.name)

    with tempfile.TemporaryDirectory(prefix="vchunk_cpp_") as tmp:
        out_base = Path(tmp) / "out"
        cmd = [
            binary, "-m", str(model_path), "-f", str(audio_path),
            "-ojf", "-of", str(out_base), "-np",
            "-l", language or "auto",
        ]
        preset = dtw_preset(model_path)
        if preset:
            # Accurate word times (token-level DTW); DTW needs flash attention off.
            cmd += ["-dtw", preset, "-nfa"]
        else:
            logger.info("No DTW preset for %s — word times come from token offsets", model_path.name)
        if prompt:
            cmd += ["--prompt", prompt]
        used_vad = False
        if vad:
            vad_path = _vad_model()
            if vad_path:
                # Keep a little padding so word edges are not clipped.
                cmd += ["--vad", "-vm", str(vad_path), "-vp", "200"]
                used_vad = True
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"whisper-cli failed ({proc.returncode}): {proc.stderr.strip()[-500:]}")
        json_path = out_base.with_suffix(".json")
        data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))

    transcript = parse_whisper_cpp_json(data)
    if used_vad and transcript.words:
        # Segment offsets are mapped back to the original timeline, token offsets are not.
        logger.info("VAD on: using segment timings (word timings are not reliable with VAD)")
        transcript.words = None
    if not transcript.text.strip():
        logger.warning("whisper.cpp returned an empty transcript — audio may be silent or too short")
    return transcript


# ── Python Whisper (CPU) ──────────────────────────────────────────────────────


def _transcribe_local(
    audio_path: Path,
    *,
    model: str = "base",
    language: str | None = None,
    prompt: str | None = None,
) -> Transcript:
    """Transcribe using local Whisper library (free, runs on-device)."""
    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "openai-whisper is not installed. Use --whisper-mode cpp (brew install whisper-cpp), "
            "or run: pip install 'video-chunker[local-whisper]'"
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
    if prompt:
        kwargs["initial_prompt"] = prompt

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
    prompt: str | None = None,
) -> Transcript:
    """Transcribe using OpenAI Whisper API."""
    logger.info("Transcribing with OpenAI Whisper API model '%s'", model)

    file_size = audio_path.stat().st_size
    if file_size <= MAX_FILE_SIZE:
        return _transcribe_openai_file(audio_path, model=model, client=client, language=language, prompt=prompt)

    logger.info("Audio is %.1f MB — splitting into chunks", file_size / 1024 / 1024)
    return _transcribe_openai_large_file(audio_path, model=model, client=client, language=language, prompt=prompt)


def _transcribe_openai_file(
    audio_path: Path,
    *,
    model: str,
    client: OpenAI,
    language: str | None = None,
    prompt: str | None = None,
) -> Transcript:
    """Transcribe a single audio file via OpenAI API."""
    kwargs: dict = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"],
    }
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt

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
    prompt: str | None = None,
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
            transcript = _transcribe_openai_file(chunk_path, model=model, client=client, language=language, prompt=prompt)
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


def timing_drift(transcript: Transcript, silences: list, duration: float) -> float:
    """Share (0–1) of transcribed speech that sits where the audio is known to be silent.

    ``silences`` come from ffmpeg silencedetect on the original audio, so speech inside
    them means Whisper's timestamps have drifted. Anything past the end of the video
    counts as drift too. Uses word timings when present, else segments.
    """
    spans = [(w.start, w.end) for w in transcript.words] if transcript.words else [
        (s.start, s.end) for s in transcript.segments
    ]
    total = sum(max(e - s, 0.0) for s, e in spans)
    if total <= 0:
        return 0.0
    bad = 0.0
    for s, e in spans:
        if e <= s:
            continue
        bad += max(0.0, e - max(s, duration))  # past the end of the video
        for si in silences:
            # Shrink each silence a little: speech edges legitimately touch it.
            a, b = si.start + 0.3, si.end - 0.3
            if b > a:
                bad += max(0.0, min(e, b) - max(s, a))
    return min(bad / total, 1.0)
