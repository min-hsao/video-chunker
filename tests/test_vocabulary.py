"""Vocabulary rules, checked against the shared cases (tests/data/vocabulary-cases.json)."""

import json
from pathlib import Path

import pytest

from video_chunker.transcriber import Transcript, TranscriptSegment, WordSegment
from video_chunker.vocabulary import Vocabulary, apply_corrections

CASES = json.loads((Path(__file__).parent / "data" / "vocabulary-cases.json").read_text(encoding="utf-8"))
VOCAB = Vocabulary.from_dict({"corrections": CASES["corrections"]})


@pytest.mark.parametrize("case", CASES["apply"], ids=lambda c: c["in"] or "<empty>")
def test_shared_apply_cases(case):
    assert VOCAB.apply(case["in"]) == case["out"]


def test_shared_keyterms_case():
    vocab = Vocabulary.from_dict(CASES["keyterms"]["vocabulary"])
    assert vocab.keyterms() == CASES["keyterms"]["out"]


def test_pure_case_fix_applied_as_written():
    assert apply_corrections("ask sam and SAM", [("sam", "Sam")]) == "ask Sam and Sam"


def test_one_pass_no_chains():
    # a→b and b→c must not turn a into c
    assert apply_corrections("alpha beta", [("alpha", "beta"), ("beta", "gamma")]) == "beta gamma"


def test_longest_from_first():
    assert apply_corrections("sam tricks", [("sam", "Sam"), ("sam tricks", "Samtrix")]) == "samtrix"


def test_later_same_from_wins_and_undo_keeps_later():
    v = Vocabulary.from_dict({"corrections": [
        {"from": "Sahm", "to": "Mean"}, {"from": "Sahm", "to": "Sam"},
        {"from": "foo", "to": "bar"}, {"from": "bar", "to": "foo"},
    ]})
    assert ("Sahm", "Sam") in v.corrections and ("Sahm", "Mean") not in v.corrections
    assert ("bar", "foo") in v.corrections and ("foo", "bar") not in v.corrections


def test_caps_and_limits():
    v = Vocabulary.from_dict({"people": [f"P{i}" for i in range(200)], "words": ["x" * 100]})
    assert len(v.people) == 100
    assert len(v.words[0]) == 60
    assert len(v.keyterms()) <= 50


def test_prompts():
    v = Vocabulary.from_dict({"my_name": "Sam", "aliases": ["Samantha"], "words": ["Samtrix"]})
    assert v.whisper_prompt() == "Sam, Samantha, Samtrix."
    line = v.names_line()
    assert 'The speaker is "Sam" (also called "Samantha").' in line
    assert '"Samtrix"' in line
    assert Vocabulary().names_line() == ""


def test_apply_to_transcript():
    t = Transcript(
        text="Sahm here",
        segments=[TranscriptSegment("Sahm here", 0, 1)],
        words=[WordSegment("Sahm", 0, 0.4), WordSegment("here", 0.4, 1)],
    )
    VOCAB.apply_to_transcript(t)
    assert t.text == "Sam here"
    assert t.segments[0].text == "Sam here"
    assert [w.word for w in t.words] == ["Sam", "here"]


def test_load(tmp_path):
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps({"my_name": "Sam", "corrections": [["Sahm", "Sam"]]}))
    v = Vocabulary.load(p)
    assert v.my_name == "Sam" and v.corrections == [("Sahm", "Sam")]
    with pytest.raises(FileNotFoundError):
        Vocabulary.load(tmp_path / "missing.json")
