"""Names, terms and spelling corrections.

A small JSON file:

    {
      "my_name": "Sam",
      "aliases": ["Samantha"],
      "people": ["Dana"],
      "words": ["Samtrix", "rocketdoc"],
      "corrections": [{"from": "Sahm", "to": "Sam"}]
    }

It is used three ways:
1. ``keyterms`` bias transcription (Whisper's initial prompt) toward the right spellings.
2. ``corrections`` fix what Whisper still gets wrong, in the transcript text, segments
   and words, before anything else (splits, AI naming, manifests) sees it.
3. ``names_line`` tells the AI how to spell names in clip descriptions.

Correction rules (shared cases in tests/data/vocabulary-cases.json):
- whole word: no letter, digit or ``_`` right before or after the match; a ``from``
  containing CJK characters matches inside text (no spaces there);
- case-insensitive match, case-preserving replacement: ALL CAPS → caps, all lower →
  lower, Capitalised → capital first letter, otherwise the target as written; a pure
  case fix ("sam" → "Sam") is applied as written;
- longest ``from`` first, one pass (no chains); a later correction with the same ``from``
  replaces the earlier; a pair that undoes another keeps only the later.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_VOCAB_PATH = Path(os.environ.get(
    "VIDEO_CHUNKER_VOCAB", Path.home() / ".config" / "video-chunker" / "vocab.json"
))

MAX_ALIASES, MAX_PEOPLE, MAX_WORDS, MAX_CORRECTIONS, MAX_TERM = 10, 100, 100, 200, 60
MAX_KEYTERMS, MAX_KEYTERM_CHARS = 50, 800

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")


def _clean_term(value) -> str:
    return str(value or "").strip()[:MAX_TERM]


@dataclass
class Vocabulary:
    my_name: str | None = None
    aliases: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    corrections: list[tuple[str, str]] = field(default_factory=list)

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        def terms(key: str, cap: int) -> list[str]:
            out = [_clean_term(t) for t in (data.get(key) or [])]
            return [t for t in out if t][:cap]

        pairs: list[tuple[str, str]] = []
        for c in (data.get("corrections") or [])[:MAX_CORRECTIONS]:
            if isinstance(c, dict):
                src, dst = _clean_term(c.get("from")), _clean_term(c.get("to"))
            elif isinstance(c, (list, tuple)) and len(c) == 2:
                src, dst = _clean_term(c[0]), _clean_term(c[1])
            else:
                continue
            if src and dst and src != dst:
                pairs.append((src, dst))

        return cls(
            my_name=_clean_term(data.get("my_name")) or None,
            aliases=terms("aliases", MAX_ALIASES),
            people=terms("people", MAX_PEOPLE),
            words=terms("words", MAX_WORDS),
            corrections=_normalise_pairs(pairs),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "Vocabulary | None":
        """Load ``path``, or the default file if it exists. None when there is none."""
        explicit = path is not None
        path = Path(path).expanduser() if path else DEFAULT_VOCAB_PATH
        if not path.exists():
            if explicit:
                raise FileNotFoundError(f"Vocabulary file not found: {path}")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Vocabulary file {path} is not valid JSON: {e}") from e
        vocab = cls.from_dict(data)
        logger.info(
            "Vocabulary %s: %d terms, %d corrections", path, len(vocab.keyterms()), len(vocab.corrections)
        )
        return vocab

    def is_empty(self) -> bool:
        return not (self.my_name or self.aliases or self.people or self.words or self.corrections)

    # ── keyterms / prompts ──────────────────────────────────────────────────

    def keyterms(self) -> list[str]:
        """Names and terms, de-duplicated ignoring case, capped at 50 terms / 800 characters."""
        candidates = (
            ([self.my_name] if self.my_name else [])
            + self.aliases + self.people + self.words
            + [dst for _, dst in self.corrections]
        )
        out: list[str] = []
        seen: set[str] = set()
        chars = 0
        for term in candidates:
            key = term.casefold()
            if not term or key in seen:
                continue
            if len(out) >= MAX_KEYTERMS or chars + len(term) > MAX_KEYTERM_CHARS:
                break
            seen.add(key)
            out.append(term)
            chars += len(term)
        return out

    def whisper_prompt(self) -> str | None:
        """Initial prompt that nudges Whisper toward these spellings."""
        terms = self.keyterms()
        return ", ".join(terms) + "." if terms else None

    def names_line(self) -> str:
        """One or two sentences for AI prompts; empty without a vocabulary."""
        parts: list[str] = []
        if self.my_name:
            also = ", ".join(f'"{a}"' for a in self.aliases)
            parts.append(
                f'The speaker is "{self.my_name}"' + (f" (also called {also})" if also else "") + "."
            )
        terms = self.keyterms()
        if terms:
            parts.append(
                "Spell these names and terms exactly as written here: "
                + ", ".join(f'"{t}"' for t in terms) + "."
            )
        return " ".join(parts)

    # ── corrections ─────────────────────────────────────────────────────────

    def apply(self, text: str) -> str:
        return apply_corrections(text, self.corrections)

    def apply_to_transcript(self, transcript):
        """Correct a Transcript in place (text, segments, words) and return it."""
        if not self.corrections:
            return transcript
        transcript.text = self.apply(transcript.text)
        for seg in transcript.segments:
            seg.text = self.apply(seg.text)
        if transcript.words:
            for w in transcript.words:
                w.word = self.apply(w.word)
        return transcript


def _normalise_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Later wins for the same ``from``; a pair undoing a later one is dropped."""
    by_from: dict[str, tuple[str, str]] = {}
    for src, dst in pairs:
        key = src.casefold()
        by_from.pop(key, None)  # re-insert so order reflects the latest
        # A later pair that undoes an earlier one (a→b then b→a) keeps only the later.
        undo_key = dst.casefold()
        if undo_key in by_from and by_from[undo_key][1].casefold() == key:
            del by_from[undo_key]
        by_from[key] = (src, dst)
    return list(by_from.values())


def _match_case(matched: str, src: str, dst: str) -> str:
    if src.casefold() == dst.casefold():
        return dst  # a pure case fix is applied as written
    letters = [c for c in matched if c.isalpha()]
    if not letters:
        return dst
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return dst.upper()
    if all(c.islower() for c in letters):
        return dst.lower()
    if letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return dst[:1].upper() + dst[1:]
    return dst


def _pattern_for(src: str) -> str:
    body = re.escape(src)
    if _CJK.search(src):
        return body
    return rf"(?<![\w]){body}(?![\w])"


def apply_corrections(text: str, corrections: list[tuple[str, str]]) -> str:
    """Apply corrections in one pass, longest ``from`` first, case-preserving."""
    if not text or not corrections:
        return text
    ordered = sorted(corrections, key=lambda p: len(p[0]), reverse=True)
    lookup = {src.casefold(): (src, dst) for src, dst in ordered}
    regex = re.compile("|".join(_pattern_for(src) for src, _ in ordered), re.IGNORECASE)

    def sub(m: re.Match) -> str:
        found = m.group(0)
        src, dst = lookup.get(found.casefold(), (found, found))
        return _match_case(found, src, dst)

    return regex.sub(sub, text)
