"""LLM-based chunk analysis using OpenAI-compatible Chat API."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from openai import OpenAI

from .grounding import UNTRUSTED_NOTE, find_segment_with_quote, normalise, quote_matches

logger = logging.getLogger(__name__)


@dataclass
class ChunkAnalysis:
    description: str
    is_complete: bool
    confidence: float
    notes: str = ""
    script_match: str = ""


@dataclass
class ChunkInfo:
    index: int
    start: float
    end: float
    duration: float
    transcript: str
    analysis: ChunkAnalysis | None = None
    # Extended metadata (added in v3)
    contact_sheet_path: str | None = None
    qc_result: object | None = None
    output_path: str = ""
    cue_triggered: bool = False
    retake: bool = False  # True if this chunk was split off as a detected retake
    metadata: object | None = None  # VideoMetadata from the source video


def analyze_chunk(
    chunk: ChunkInfo,
    *,
    video_type: str = "product-demo",
    script: str | None = None,
    model: str = "deepseek-chat",
    client: OpenAI | None = None,
    recording_context: str = "",
    names_line: str = "",
) -> ChunkAnalysis:
    """Analyze a chunk transcript using an LLM to determine completeness and description."""
    if client is None:
        client = OpenAI()

    system_prompt = f"""You are a video production assistant analyzing transcript chunks from a {video_type} recording session.

For each chunk, determine:
1. A brief description (3-6 words, lowercase, suitable for a filename)
2. Whether this appears to be a complete take or an incomplete/aborted one
3. Your confidence level (0.0-1.0)
4. Any notes about the content

Respond in JSON format only:
{{
    "description": "brief description here",
    "is_complete": true,
    "confidence": 0.95,
    "notes": "optional notes"
}}"""

    if script:
        system_prompt += f"""

The creator is working from this draft script:
---
{script}
---

Also assess how well this chunk matches the script and include a "script_match" field describing the alignment."""

    if recording_context:
        system_prompt += f"""

Recording context (from camera metadata):
{recording_context}

Use this context to make descriptions more specific and informative when relevant
(e.g. include location or camera in the description if it adds value).
Do NOT force location/date into every description — only when it genuinely helps identify the clip."""

    if names_line:
        system_prompt += f"\n\n{names_line}"
    system_prompt += f"\n\n{UNTRUSTED_NOTE}"

    user_prompt = f"""Analyze this transcript chunk (chunk #{chunk.index + 1}, duration: {chunk.duration:.1f}s):

---
{chunk.transcript or "(no speech detected)"}
---"""

    logger.info("Analyzing chunk %d with model %s", chunk.index + 1, model)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        content = response.choices[0].message.content or "{}"
    except Exception as e:
        logger.warning("LLM analysis failed for chunk %d: %s", chunk.index + 1, e)
        content = "{}"

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON for chunk %d", chunk.index + 1)
        data = {}

    return ChunkAnalysis(
        description=data.get("description", "untitled_segment"),
        is_complete=bool(data.get("is_complete", False)),
        confidence=float(data.get("confidence", 0.5)),
        notes=str(data.get("notes", "")),
        script_match=str(data.get("script_match", "")),
    )


def analyze_chunks(
    chunks: list[ChunkInfo],
    *,
    video_type: str = "product-demo",
    script: str | None = None,
    model: str = "deepseek-chat",
    client: OpenAI | None = None,
    recording_context: str = "",
    names_line: str = "",
) -> list[ChunkInfo]:
    """Analyze all chunks and attach analysis results."""
    if client is None:
        client = OpenAI()

    for chunk in chunks:
        chunk.analysis = analyze_chunk(
            chunk,
            video_type=video_type,
            script=script,
            model=model,
            client=client,
            recording_context=recording_context,
            names_line=names_line,
        )

    return chunks


# ── Smart AI features ────────────────────────────────────────────────────────
#
# Every request numbers its items with a batch-local "id" and every answer is looked up by
# that id (earlier versions mixed absolute and batch-relative indexes, so after the first
# batch decisions landed on the wrong segments). Decisions that remove footage must quote
# the words they are about; grounding.quote_matches checks the quote against the segment
# and drops the decision if it is not there.


def _chat_json(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content or "{}")


def _by_id(items: list, key: str = "decisions") -> dict[int, dict]:
    out: dict[int, dict] = {}
    for d in items or []:
        if not isinstance(d, dict):
            continue
        raw = d.get("id", d.get("index"))
        try:
            out[int(raw)] = d
        except (TypeError, ValueError):
            continue
    return out


def _system(prompt: str, names_line: str = "") -> str:
    extra = f"\n\n{names_line}" if names_line else ""
    return f"{prompt}\n\n{UNTRUSTED_NOTE}{extra}"


def _to_float(value, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_transcript_window(
    transcript_text: str,
    segments: list,
    time: float,
    window_before: float = 30.0,
    window_after: float = 30.0,
) -> tuple[str, str]:
    """Get transcript context around a timestamp.

    Returns (text_before, text_after) split at the given time.
    """
    before_parts: list[str] = []
    after_parts: list[str] = []

    for seg in segments:
        if seg.end <= time and (time - seg.end) <= window_before:
            before_parts.append(seg.text)
        elif seg.start >= time and (seg.start - time) <= window_after:
            after_parts.append(seg.text)
        elif seg.start < time < seg.end:
            # Segment straddles the split point
            before_parts.append(seg.text)

    return (
        " ".join(before_parts) if before_parts else "(start of video)",
        " ".join(after_parts) if after_parts else "(end of video)",
    )


@dataclass
class SplitValidation:
    """LLM verdict on whether a candidate split point is a good cut."""
    timestamp: float
    approved: bool
    reason: str
    confidence: float


def validate_split_points(
    candidate_times: list[float],
    *,
    transcript_text: str,
    segments: list,
    model: str = "deepseek-chat",
    client: OpenAI | None = None,
    batch_size: int = 15,
    names_line: str = "",
) -> list[SplitValidation]:
    """Use LLM to validate candidate split points.

    For each silence, asks the LLM whether it's a natural topic boundary
    (good split) or a mid-thought pause (bad split). Unanswered candidates are kept.
    """
    if not candidate_times:
        return []
    if client is None:
        client = OpenAI()

    results: list[SplitValidation] = []
    system_prompt = _system("""You are a video editor deciding where to split a long recording into clips.

For each candidate split point, you see transcript text BEFORE and AFTER the proposed cut.
Decide whether this is a GOOD or BAD place to split.

GOOD splits: topic changes, completed thoughts, natural paragraph breaks.
BAD splits: mid-sentence pauses, speaker thinking mid-idea, brief hesitation in flow.

Answer every candidate, using its "id". Respond in JSON only:
{"decisions": [{"id": 0, "approved": true, "reason": "brief explanation", "confidence": 0.9}]}""", names_line)

    for batch_start in range(0, len(candidate_times), batch_size):
        batch = candidate_times[batch_start:batch_start + batch_size]
        candidates_info = []
        for k, t in enumerate(batch):
            before, after = _get_transcript_window(transcript_text, segments, t)
            candidates_info.append({
                "id": k,
                "timestamp": round(t, 1),
                "text_before": before[-500:],
                "text_after": after[:500],
            })

        logger.info("Validating %d split points with LLM (batch %d-%d)",
                    len(batch), batch_start, batch_start + len(batch))
        try:
            data = _chat_json(client, model, system_prompt,
                              f"Evaluate these {len(batch)} candidate split points:\n\n"
                              + json.dumps(candidates_info, indent=2, ensure_ascii=False))
            decisions = _by_id(data.get("decisions"))
        except Exception as e:
            logger.warning("LLM split validation failed: %s — keeping all candidates", e)
            decisions = {}

        for k, t in enumerate(batch):
            dec = decisions.get(k)
            if dec is None:
                results.append(SplitValidation(t, True, "no answer — kept", 0.5))
                continue
            results.append(SplitValidation(
                timestamp=t,
                approved=bool(dec.get("approved", True)),
                reason=str(dec.get("reason", "")),
                confidence=_to_float(dec.get("confidence")),
            ))

    approved = sum(1 for r in results if r.approved)
    logger.info("Split validation: %d/%d approved", approved, len(results))
    return results


@dataclass
class RetakeDetection:
    """LLM verdict on whether a segment is a retake of the one before it.

    ``timestamp`` is where the new attempt starts (a split point in chunk mode);
    ``prev_start``/``prev_end`` is the abandoned attempt (what clean mode removes).
    """
    timestamp: float
    is_retake: bool
    reason: str
    confidence: float
    prev_start: float = 0.0
    prev_end: float = 0.0
    quote: str = ""
    grounded: bool = False


def detect_retakes_llm(
    segments: list,
    *,
    model: str = "deepseek-chat",
    client: OpenAI | None = None,
    batch_size: int = 20,
    names_line: str = "",
) -> list[RetakeDetection]:
    """Use LLM to detect retakes that string similarity misses.

    Catches verbal cues ("let me try that again"), the same idea in different words,
    and abandoned mid-thought segments. A retake counts only when the model quotes words
    that really are in the earlier (abandoned) segment.
    """
    if len(segments) < 2:
        return []
    if client is None:
        client = OpenAI()

    pairs: list[tuple[object, object]] = []
    for i in range(1, len(segments)):
        prev, curr = segments[i - 1], segments[i]
        if curr.start - prev.start <= 60.0:
            pairs.append((prev, curr))
    if not pairs:
        return []

    system_prompt = _system("""You are a video editor detecting retakes in a recording session.

For each pair of consecutive transcript segments, decide if the SECOND segment is a RETAKE
of the FIRST: the speaker is restarting, redoing or re-attempting what they just said, so
the first segment is the abandoned attempt.

Signs of a retake:
- Verbal cues: "let me try again", "one more time", "starting over", "let me redo that"
- The second segment covers the same idea as the first (even in different words)
- The first segment is abandoned mid-thought
- The second segment starts the same topic more clearly

NOT a retake: continuing the thought, the next point, answering a different question.

For every retake, "quote" must be words copied exactly from the FIRST segment (the
abandoned attempt). Answer every pair, using its "id". Respond in JSON only:
{"decisions": [{"id": 0, "is_retake": true, "quote": "words from the first segment", "reason": "brief", "confidence": 0.9}]}""", names_line)

    results: list[RetakeDetection] = []
    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start:batch_start + batch_size]
        payload = [
            {
                "id": k,
                "first_segment": {"text": prev.text, "start": round(prev.start, 1), "end": round(prev.end, 1)},
                "second_segment": {"text": curr.text, "start": round(curr.start, 1), "end": round(curr.end, 1)},
            }
            for k, (prev, curr) in enumerate(batch)
        ]
        logger.info("Detecting retakes via LLM: %d pairs (batch %d-%d)",
                    len(batch), batch_start, batch_start + len(batch))
        try:
            data = _chat_json(client, model, system_prompt,
                              f"Evaluate these {len(batch)} segment pairs for retakes:\n\n"
                              + json.dumps(payload, indent=2, ensure_ascii=False))
            decisions = _by_id(data.get("decisions"))
        except Exception as e:
            logger.warning("LLM retake detection failed: %s", e)
            continue

        for k, (prev, curr) in enumerate(batch):
            dec = decisions.get(k, {})
            is_retake = bool(dec.get("is_retake", False))
            quote = str(dec.get("quote", "") or "")
            reason = str(dec.get("reason", "no answer"))
            grounded = is_retake and quote_matches(quote, prev.text)
            if is_retake and not grounded:
                logger.info("Retake at %.1fs dropped: quote %r is not in the earlier segment", curr.start, quote)
                reason = f"dropped (quote not in transcript): {reason}"
            results.append(RetakeDetection(
                timestamp=curr.start,
                is_retake=grounded,
                reason=reason,
                confidence=_to_float(dec.get("confidence")),
                prev_start=prev.start,
                prev_end=prev.end,
                quote=quote,
                grounded=grounded,
            ))

    logger.info("LLM retake detection: %d/%d pairs are grounded retakes",
                sum(1 for r in results if r.is_retake), len(results))
    return results


@dataclass
class FillerDetection:
    """LLM verdict on filler speech.

    ``start``/``end`` is the time to remove: just the quoted filler words when word
    timings locate them, otherwise the whole segment, and only when the whole segment
    is the filler.
    """
    start: float
    end: float
    is_filler: bool
    reason: str
    confidence: float
    quote: str = ""
    grounded: bool = False


# Potential filler words/phrases to flag for LLM review
_POTENTIAL_FILLERS = {
    "um", "uh", "ah", "er", "hmm", "huh", "mm",
    "you know", "like", "I mean", "so yeah",
    "basically", "literally", "actually",
    "kind of", "sort of", "right",
}


def locate_quote_span(quote: str, words: list | None, start: float, end: float) -> tuple[float, float] | None:
    """Time span of the words that spell ``quote`` inside [start, end], or None."""
    target = normalise(quote)
    if not target or not words:
        return None
    inside = [w for w in words if w.end > start - 0.25 and w.start < end + 0.25]
    norm = [normalise(w.word) for w in inside]
    for i in range(len(inside)):
        acc = ""
        for j in range(i, len(inside)):
            acc += norm[j]
            if acc == target:
                return inside[i].start, inside[j].end
            if not target.startswith(acc):
                break
    return None


def detect_fillers_llm(
    segments: list,
    *,
    model: str = "deepseek-chat",
    client: OpenAI | None = None,
    batch_size: int = 25,
    words: list | None = None,
    names_line: str = "",
) -> list[FillerDetection]:
    """Use LLM to detect filler words in context.

    Unlike the hardcoded list, this understands that "you know" is filler in
    "you know, like, whatever" but not in "you know the thing about AI is...".
    Only the quoted filler is removed (located with word timings); without word
    timings a segment is removed only when it is nothing but the filler.
    """
    if not segments:
        return []
    if client is None:
        client = OpenAI()

    candidates = []
    for seg in segments:
        text_lower = seg.text.lower().strip().rstrip(".,!?;:")
        word_count = len(text_lower.split())
        if word_count <= 2 or any(re.search(rf"(?<!\w){re.escape(f.lower())}(?!\w)", text_lower) for f in _POTENTIAL_FILLERS):
            candidates.append(seg)
    if not candidates:
        return []

    system_prompt = _system("""You are a video editor identifying filler words and phrases in speech transcripts.

For each segment decide whether it contains FILLER (a meaningless verbal tic) that can be
removed without changing the meaning.

FILLER: "um" / "uh" on their own; "you know" or "like" as a tic ("it was like, you know,
like really good"); "I mean" as an empty bridge; "basically" / "literally" / "actually"
adding nothing.
MEANINGFUL (not filler): "you know the thing about AI is..."; "I like that approach";
"I mean what I say"; "actually, that's correct" (a real correction).

For each filler, "quote" must be the filler words copied exactly from the segment, and only
those words (for example "um, you know"). Answer every segment, using its "id". JSON only:
{"decisions": [{"id": 0, "is_filler": true, "quote": "um", "reason": "brief", "confidence": 0.9}]}""", names_line)

    results: list[FillerDetection] = []
    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start:batch_start + batch_size]
        payload = [
            {"id": k, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
            for k, s in enumerate(batch)
        ]
        logger.info("Detecting fillers via LLM: %d segments (batch %d-%d)",
                    len(batch), batch_start, batch_start + len(batch))
        try:
            data = _chat_json(client, model, system_prompt,
                              f"Evaluate these {len(batch)} segments for filler:\n\n"
                              + json.dumps(payload, indent=2, ensure_ascii=False))
            decisions = _by_id(data.get("decisions"))
        except Exception as e:
            logger.warning("LLM filler detection failed: %s", e)
            continue

        for k, seg in enumerate(batch):
            dec = decisions.get(k, {})
            is_filler = bool(dec.get("is_filler", False))
            quote = str(dec.get("quote", "") or "")
            reason = str(dec.get("reason", "no answer"))
            span: tuple[float, float] | None = None
            if is_filler and quote_matches(quote, seg.text):
                span = locate_quote_span(quote, words, seg.start, seg.end)
                if span is None and normalise(quote) == normalise(seg.text):
                    span = (seg.start, seg.end)  # the whole segment is the filler
            if is_filler and span is None:
                logger.info("Filler at %.1fs dropped: quote %r not located in %r", seg.start, quote, seg.text)
                reason = f"dropped (quote not located): {reason}"
            results.append(FillerDetection(
                start=span[0] if span else seg.start,
                end=span[1] if span else seg.end,
                is_filler=span is not None,
                reason=reason,
                confidence=_to_float(dec.get("confidence")),
                quote=quote,
                grounded=span is not None,
            ))

    logger.info("LLM filler detection: %d/%d segments have grounded filler",
                sum(1 for r in results if r.is_filler), len(results))
    return results


@dataclass
class ContentCutSuggestion:
    """LLM suggestion for content to cut in clean mode."""
    start: float
    end: float
    reason: str  # "tangent", "redundant", "thinking_aloud", "off_topic"
    confidence: float
    quote: str = ""


def suggest_content_cuts(
    segments: list,
    *,
    model: str = "deepseek-chat",
    client: OpenAI | None = None,
    names_line: str = "",
) -> list[ContentCutSuggestion]:
    """Use LLM to identify whole segments to cut in clean mode.

    Detects tangents, redundant repetition, thinking aloud and off-topic chatter.
    Each cut must quote its segment; a quote found under a different segment (8+ letters)
    is re-pointed there, otherwise the cut is dropped.
    """
    if not segments:
        return []
    if client is None:
        client = OpenAI()

    system_prompt = _system("""You are editing a video recording into a clean, polished final cut.

Review the transcript segments and identify any that should be REMOVED:
- Tangents: an unrelated detour before returning to the topic
- Redundant: repeats a point already made (not a retake, just repetitive)
- Off-topic chatter: casual talk, side comments, setup chatter
- Thinking aloud: verbalised thinking that adds nothing

Do NOT cut core content, explanations, demonstrations, natural transitions, normal pauses,
or humour and personality that add to the video.

For each cut, "quote" must be words copied exactly from that segment. JSON only:
{"cuts": [{"id": 5, "reason": "tangent|redundant|off_topic|thinking_aloud", "quote": "words from the segment", "confidence": 0.9}]}
If nothing should be cut, return {"cuts": []}""", names_line)

    window_size = 40
    suggestions: list[ContentCutSuggestion] = []
    for win_start in range(0, len(segments), window_size // 2):
        window = segments[win_start:win_start + window_size]
        if len(window) < 3:
            continue
        payload = [
            {"id": k, "start": round(s.start, 1), "end": round(s.end, 1), "text": s.text}
            for k, s in enumerate(window)
        ]
        logger.info("LLM content review: %d segments (window %d-%d)",
                    len(window), win_start, win_start + len(window))
        try:
            data = _chat_json(client, model, system_prompt,
                              "Review these transcript segments for cutting:\n\n"
                              + json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.warning("LLM content cut suggestion failed: %s", e)
            continue

        for k, cut in _by_id(data.get("cuts")).items():
            quote = str(cut.get("quote", "") or "")
            seg = window[k] if 0 <= k < len(window) else None
            if seg is None or not quote_matches(quote, seg.text):
                moved = find_segment_with_quote(quote, window)
                if moved is None:
                    logger.info("Content cut dropped: quote %r not found", quote)
                    continue
                seg = window[moved]
            suggestions.append(ContentCutSuggestion(
                start=seg.start,
                end=seg.end,
                reason=str(cut.get("reason", "unspecified")),
                confidence=_to_float(cut.get("confidence")),
                quote=quote,
            ))

    seen: set[tuple[float, float]] = set()
    deduped: list[ContentCutSuggestion] = []
    for s in suggestions:
        key = (round(s.start, 1), round(s.end, 1))
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    logger.info("LLM content review: %d grounded cut suggestions", len(deduped))
    return deduped
