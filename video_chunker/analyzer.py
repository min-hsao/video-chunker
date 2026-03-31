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
    output_path: str = ""
    cue_triggered: bool = False


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
