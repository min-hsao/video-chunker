"""LLM-based chunk analysis using OpenAI-compatible Chat API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

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
    output_path: str | None = None
    output_path: str = ""
    cue_triggered: bool = False
    retake: bool = False  # True if this chunk was split off as a detected retake


def analyze_chunk(
    chunk: ChunkInfo,
    *,
    video_type: str = "product-demo",
    script: str | None = None,
    model: str = "deepseek-chat",
    client: OpenAI | None = None,
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
        )

    return chunks


# ── Smart AI features ────────────────────────────────────────────────────────


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
) -> list[SplitValidation]:
    """Use LLM to validate candidate split points.

    For each silence, asks the LLM whether it's a natural topic boundary
    (good split) or a mid-thought pause (bad split).

    Batches multiple candidates per API call to reduce costs.
    """
    if not candidate_times:
        return []
    if client is None:
        client = OpenAI()

    results: list[SplitValidation] = []

    # Process in batches
    for batch_start in range(0, len(candidate_times), batch_size):
        batch = candidate_times[batch_start:batch_start + batch_size]

        # Build context for each candidate
        candidates_info: list[dict] = []
        for i, t in enumerate(batch):
            before, after = _get_transcript_window(transcript_text, segments, t)
            candidates_info.append({
                "index": i,
                "timestamp": round(t, 1),
                "text_before": before[-500:],  # Cap context length
                "text_after": after[:500],
            })

        system_prompt = """You are a video editor deciding where to split a long recording into clips.

For each candidate split point, you see transcript text BEFORE and AFTER the proposed cut.
Decide whether this is a GOOD or BAD place to split.

GOOD splits: topic changes, completed thoughts, natural paragraph breaks.
BAD splits: mid-sentence pauses, speaker thinking mid-idea, brief hesitation in flow.

Respond in JSON format only:
{
    "decisions": [
        {
            "index": 0,
            "approved": true,
            "reason": "Brief explanation",
            "confidence": 0.9
        },
        ...
    ]
}"""

        user_prompt = f"""Evaluate these {len(batch)} candidate split points:

{json.dumps(candidates_info, indent=2)}"""

        logger.info("Validating %d split points with LLM (batch %d-%d)",
                     len(batch), batch_start, batch_start + len(batch))

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            decisions = {d["index"]: d for d in data.get("decisions", [])}
        except Exception as e:
            logger.warning("LLM split validation failed: %s — keeping all candidates", e)
            # On failure, keep all candidates (safe fallback)
            for t in batch:
                results.append(SplitValidation(
                    timestamp=t, approved=True,
                    reason=f"LLM unavailable: {e}", confidence=0.5,
                ))
            continue

        for i, t in enumerate(batch):
            dec = decisions.get(i, {})
            results.append(SplitValidation(
                timestamp=t,
                approved=bool(dec.get("approved", True)),
                reason=str(dec.get("reason", "no response")),
                confidence=float(dec.get("confidence", 0.5)),
            ))

    approved = sum(1 for r in results if r.approved)
    logger.info("Split validation: %d/%d approved", approved, len(results))
    return results


@dataclass
class RetakeDetection:
    """LLM verdict on whether a segment is a retake/restart."""
    timestamp: float
    is_retake: bool
    reason: str
    confidence: float


def detect_retakes_llm(
    segments: list,
    *,
    model: str = "deepseek-chat",
    client: OpenAI | None = None,
    batch_size: int = 20,
) -> list[RetakeDetection]:
    """Use LLM to detect retakes that string similarity misses.

    Catches:
    - Verbal cues ("let me try that again", "one more time")
    - Same idea with different words
    - Abandoned mid-thought segments
    """
    if len(segments) < 2:
        return []
    if client is None:
        client = OpenAI()

    results: list[RetakeDetection] = []

    # Build pairs of consecutive segments for batch evaluation
    pairs: list[dict] = []
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]
        # Only check within 60s window
        if curr.start - prev.start > 60.0:
            continue
        pairs.append({
            "index": i,
            "timestamp": round(curr.start, 1),
            "prev_segment": {
                "text": prev.text,
                "start": round(prev.start, 1),
                "end": round(prev.end, 1),
            },
            "curr_segment": {
                "text": curr.text,
                "start": round(curr.start, 1),
                "end": round(curr.end, 1),
            },
        })

    if not pairs:
        return []

    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start:batch_start + batch_size]

        system_prompt = """You are a video editor detecting retakes in a recording session.

For each pair of consecutive transcript segments, decide if the second segment is a RETAKE
of the first. A retake means the speaker is restarting, redoing, or re-attempting what
they just said.

Signs of a retake:
- Verbal cues: "let me try again", "one more time", "starting over", "let me redo that"
- The second segment covers the same topic/idea as the first (even with different words)
- The first segment appears abandoned mid-thought (incomplete sentence)
- The second segment starts the same topic but more clearly/confidently

NOT a retake:
- Continuing the same thought
- Natural follow-up or next point
- Answering a different question

Respond in JSON format only:
{
    "decisions": [
        {
            "index": 0,
            "is_retake": false,
            "reason": "Brief explanation",
            "confidence": 0.9
        }
    ]
}"""

        user_prompt = f"""Evaluate these {len(batch)} segment pairs for retakes:

{json.dumps(batch, indent=2)}"""

        logger.info("Detecting retakes via LLM: %d pairs (batch %d-%d)",
                     len(batch), batch_start, batch_start + len(batch))

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            decisions = {d["index"]: d for d in data.get("decisions", [])}
        except Exception as e:
            logger.warning("LLM retake detection failed: %s", e)
            continue

        for pair in batch:
            i = pair["index"]
            dec = decisions.get(i - pairs[0]["index"], {})
            # decisions indexed relative to batch
            batch_idx = i - pairs[0]["index"]
            dec = decisions.get(batch_idx, {})
            results.append(RetakeDetection(
                timestamp=pair["timestamp"],
                is_retake=bool(dec.get("is_retake", False)),
                reason=str(dec.get("reason", "no response")),
                confidence=float(dec.get("confidence", 0.5)),
            ))

    retake_count = sum(1 for r in results if r.is_retake)
    logger.info("LLM retake detection: %d/%d segments identified as retakes",
                 retake_count, len(results))
    return results


@dataclass
class FillerDetection:
    """LLM verdict on whether a segment contains filler vs meaningful speech."""
    start: float
    end: float
    is_filler: bool
    reason: str
    confidence: float


# Potential filler words/phrases to flag for LLM review
_POTENTIAL_FILLERS = {
    "um", "uh", "ah", "er", "hmm", "huh", "mm",
    "you know", "like", "I mean", "so yeah",
    "basically", "literally", "actually",
    "kind of", "sort of", "right",
}


def detect_fillers_llm(
    segments: list,
    *,
    model: str = "deepseek-chat",
    client: OpenAI | None = None,
    batch_size: int = 25,
) -> list[FillerDetection]:
    """Use LLM to intelligently detect filler words in context.

    Unlike the hardcoded list, this understands that "you know" is filler
    in "you know, like, whatever" but NOT in "you know the thing about AI is..."
    """
    if not segments:
        return []
    if client is None:
        client = OpenAI()

    # Pre-filter: only send segments that contain potential filler words
    candidates: list[dict] = []
    for i, seg in enumerate(segments):
        text_lower = seg.text.lower().strip().rstrip(".,!?;:")
        # Check if any potential filler appears in this segment
        has_potential = any(
            f in text_lower
            for f in _POTENTIAL_FILLERS
        )
        # Also check very short segments (single word = likely filler)
        word_count = len(text_lower.split())
        if has_potential or word_count <= 2:
            candidates.append({
                "index": i,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text,
                "word_count": word_count,
            })

    if not candidates:
        return []

    results: list[FillerDetection] = []

    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start:batch_start + batch_size]

        system_prompt = """You are a video editor identifying filler words and phrases in speech transcripts.

For each segment, decide if it is FILLER (meaningless verbal tic) or MEANINGFUL (part of the content).

Examples of FILLER:
- "um" standing alone
- "you know" as a verbal tic ("um, you know, like...")
- "like" used as filler ("it was like, you know, like really good")
- "I mean" as a bridge with no content
- "basically" / "literally" / "actually" when adding no information

Examples of MEANINGFUL (NOT filler):
- "you know the thing about AI is..." ("you know" introduces a topic)
- "I like that approach" ("like" is a verb)
- "I mean what I say" (literal use)
- "actually, that's correct" (adds correction/emphasis with substance)
- Short but content-bearing words in context

Respond in JSON format only:
{
    "decisions": [
        {
            "index": 0,
            "is_filler": true,
            "reason": "Brief explanation",
            "confidence": 0.9
        }
    ]
}"""

        user_prompt = f"""Evaluate these {len(batch)} segments for filler vs meaningful speech:

{json.dumps(batch, indent=2)}"""

        logger.info("Detecting fillers via LLM: %d segments (batch %d-%d)",
                     len(batch), batch_start, batch_start + len(batch))

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            decisions = {d["index"]: d for d in data.get("decisions", [])}
        except Exception as e:
            logger.warning("LLM filler detection failed: %s", e)
            continue

        for cand in batch:
            batch_idx = cand["index"] - batch[0]["index"]
            dec = decisions.get(batch_idx, {})
            results.append(FillerDetection(
                start=cand["start"],
                end=cand["end"],
                is_filler=bool(dec.get("is_filler", False)),
                reason=str(dec.get("reason", "no response")),
                confidence=float(dec.get("confidence", 0.5)),
            ))

    filler_count = sum(1 for r in results if r.is_filler)
    logger.info("LLM filler detection: %d/%d segments identified as filler",
                 filler_count, len(results))
    return results


@dataclass
class ContentCutSuggestion:
    """LLM suggestion for content to cut in clean mode."""
    start: float
    end: float
    reason: str  # "tangent", "redundant", "filler_pause", "off_topic"
    confidence: float


def suggest_content_cuts(
    segments: list,
    *,
    model: str = "deepseek-chat",
    client: OpenAI | None = None,
) -> list[ContentCutSuggestion]:
    """Use LLM to identify content that should be cut in clean mode.

    Detects:
    - Tangents and off-topic rants
    - Redundant repetitions (same thing said multiple times)
    - Long thinking pauses that aren't dramatic pauses
    - Off-topic chatter between takes
    """
    if not segments:
        return []
    if client is None:
        client = OpenAI()

    # Build a timeline of all segments
    timeline: list[dict] = []
    for i, seg in enumerate(segments):
        timeline.append({
            "index": i,
            "start": round(seg.start, 1),
            "end": round(seg.end, 1),
            "text": seg.text,
        })

    # If too many segments, split into overlapping windows
    window_size = 40
    all_suggestions: list[ContentCutSuggestion] = []

    for win_start in range(0, len(timeline), window_size // 2):
        window = timeline[win_start:win_start + window_size]
        if len(window) < 3:
            continue

        system_prompt = """You are editing a video recording to create a clean, polished final cut.

Review the transcript segments and identify any that should be REMOVED:
- Tangents: speaker goes off on an unrelated tangent before returning to topic
- Redundant: speaker repeats the same point they already made (not a retake, just repetitive)
- Off-topic chatter: casual talk, side comments, setup chatter not part of the content
- Thinking aloud: verbalized thinking that doesn't add value ("hmm, let me think about that...")

Do NOT cut:
- Core content, explanations, demonstrations
- Natural transitions between topics
- Brief pauses or normal speech flow
- Humor or personality that adds to the video

Respond in JSON format only:
{
    "cuts": [
        {
            "index": 5,
            "reason": "tangent|redundant|off_topic|thinking_aloud",
            "confidence": 0.9,
            "explanation": "Brief explanation"
        }
    ]
}

If nothing should be cut, return {"cuts": []}"""

        user_prompt = f"""Review these transcript segments for cutting:

{json.dumps(window, indent=2)}"""

        logger.info("LLM content review: %d segments (window %d-%d)",
                     len(window), win_start, win_start + len(window))

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
        except Exception as e:
            logger.warning("LLM content cut suggestion failed: %s", e)
            continue

        for cut in data.get("cuts", []):
            idx = cut.get("index")
            if idx is None or idx < 0 or idx >= len(window):
                continue
            actual_idx = win_start + idx
            if actual_idx >= len(segments):
                continue
            seg = segments[actual_idx]
            all_suggestions.append(ContentCutSuggestion(
                start=seg.start,
                end=seg.end,
                reason=str(cut.get("reason", "unspecified")),
                confidence=float(cut.get("confidence", 0.5)),
            ))

    # Deduplicate overlapping suggestions
    seen: set[tuple[float, float]] = set()
    deduped: list[ContentCutSuggestion] = []
    for s in all_suggestions:
        key = (round(s.start, 1), round(s.end, 1))
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    logger.info("LLM content review: %d cut suggestions", len(deduped))
    return deduped
