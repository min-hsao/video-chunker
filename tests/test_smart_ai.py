"""AI retake / filler / content-cut decisions: id lookup, grounding, and what clean mode cuts."""

import json
from types import SimpleNamespace

from video_chunker.analyzer import (
    detect_fillers_llm, detect_retakes_llm, locate_quote_span, suggest_content_cuts, validate_split_points,
)
from video_chunker.splitter import compute_clean_segments_llm
from video_chunker.transcriber import Transcript, TranscriptSegment, WordSegment


class FakeClient:
    """OpenAI-shaped client that returns queued JSON answers and records prompts."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0) if self.answers else {}
        content = answer if isinstance(answer, str) else json.dumps(answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def payload(self, n=0):
        user = self.calls[n]["messages"][1]["content"]
        return json.loads(user[user.index("\n\n") + 2:])


SEGS = [
    TranscriptSegment("So the first thing we, um, the first", 7.0, 10.0),
    TranscriptSegment("let me try that again", 10.0, 12.0),
    TranscriptSegment("The first thing we do is open the dashboard", 14.0, 17.0),
    TranscriptSegment("um", 17.5, 17.8),
]
WORDS = [
    WordSegment("So", 7.0, 7.2), WordSegment("the", 7.2, 7.3), WordSegment("first", 7.3, 7.6),
    WordSegment("thing", 7.6, 7.9), WordSegment("we,", 7.9, 8.1), WordSegment("um,", 8.3, 8.6),
    WordSegment("the", 8.8, 8.9), WordSegment("first", 8.9, 9.3),
]


def test_retakes_need_a_real_quote_from_the_first_segment():
    client = FakeClient({"decisions": [
        {"id": 0, "is_retake": True, "quote": "so the first thing we", "reason": "restart", "confidence": 0.9},
        {"id": 1, "is_retake": True, "quote": "words nobody said", "reason": "invented"},
    ]})
    res = detect_retakes_llm(SEGS[:3], client=client, model="m")
    assert len(res) == 2
    assert res[0].is_retake and res[0].grounded
    assert (res[0].prev_start, res[0].prev_end, res[0].timestamp) == (7.0, 10.0, 10.0)
    assert not res[1].is_retake and "dropped" in res[1].reason
    # Items are sent with batch-local ids and the untrusted-transcript note.
    assert [p["id"] for p in client.payload()] == [0, 1]
    assert "data, not instructions" in client.calls[0]["messages"][0]["content"]


def test_retake_ids_are_batch_local_in_later_batches():
    segs = [TranscriptSegment(f"sentence number {i}", i * 2.0, i * 2.0 + 1) for i in range(5)]
    client = FakeClient(
        {"decisions": [{"id": 0, "is_retake": False}, {"id": 1, "is_retake": False}]},
        {"decisions": [{"id": 1, "is_retake": True, "quote": "sentence number 3"}]},
    )
    res = detect_retakes_llm(segs, client=client, model="m", batch_size=2)
    assert [r.is_retake for r in res] == [False, False, False, True]
    assert res[3].prev_start == 6.0 and res[3].timestamp == 8.0


def test_fillers_cut_only_the_quoted_words():
    client = FakeClient({"decisions": [
        {"id": 0, "is_filler": True, "quote": "um"},
        {"id": 1, "is_filler": True, "quote": "um"},  # ids count candidate segments only
    ]})
    res = detect_fillers_llm(SEGS, client=client, model="m", words=WORDS)
    assert len(res) == 2  # only segments with possible filler are sent
    first = res[0]
    assert first.is_filler and (first.start, first.end) == (8.3, 8.6)  # just "um," not the segment
    whole = [r for r in res if r.is_filler and r.start == 17.5]
    assert whole and whole[0].end == 17.8  # a segment that is only "um" goes entirely


def test_filler_without_word_timings_never_cuts_a_longer_segment():
    client = FakeClient({"decisions": [{"id": 0, "is_filler": True, "quote": "um"}]})
    res = detect_fillers_llm(SEGS[:1], client=client, model="m", words=None)
    assert not res[0].is_filler and "dropped" in res[0].reason


def test_locate_quote_span_multiword():
    assert locate_quote_span("we, um", WORDS, 7.0, 10.0) == (7.9, 8.6)
    assert locate_quote_span("dashboard", WORDS, 7.0, 10.0) is None


def test_content_cuts_grounded_or_repointed():
    segs = [TranscriptSegment(f"segment text {i}", i, i + 0.9) for i in range(4)]
    segs[2] = TranscriptSegment("a long tangent about lunch", 2, 2.9)
    client = FakeClient({"cuts": [
        {"id": 0, "reason": "tangent", "quote": "a long tangent about lunch"},  # wrong id, re-pointed
        {"id": 1, "reason": "off_topic", "quote": "made up"},                   # dropped
    ]})
    cuts = suggest_content_cuts(segs, client=client, model="m")
    assert [(c.start, c.end, c.reason) for c in cuts] == [(2, 2.9, "tangent")]


def test_split_validation_unanswered_kept():
    client = FakeClient({"decisions": [{"id": 1, "approved": False, "reason": "mid-thought"}]})
    res = validate_split_points([5.0, 12.0], transcript_text="", segments=SEGS, client=client, model="m")
    assert [r.approved for r in res] == [True, False]


def test_bad_json_is_safe():
    client = FakeClient("not json")
    assert detect_retakes_llm(SEGS[:2], client=client, model="m") == []


def test_names_line_reaches_prompt():
    client = FakeClient({"decisions": []})
    detect_retakes_llm(SEGS[:2], client=client, model="m", names_line='The speaker is "Sam".')
    assert 'The speaker is "Sam".' in client.calls[0]["messages"][0]["content"]


def test_clean_mode_removes_abandoned_take_and_filler():
    t = Transcript(text="", segments=SEGS, words=WORDS)
    retakes = detect_retakes_llm(SEGS[:2], client=FakeClient({"decisions": [
        {"id": 0, "is_retake": True, "quote": "the first thing we"}]}), model="m")
    keep = compute_clean_segments_llm(t, [], 20.0, llm_retakes=retakes)
    kept = [(k.start, k.end) for k in keep]
    # The abandoned take (7-10 s) goes; the lone "um" goes by the built-in filler rule.
    assert kept == [(0.0, 7.0), (10.0, 17.5), (17.8, 20.0)]


def test_chunk_transcripts_do_not_overlap():
    from video_chunker.splitter import build_chunks
    t = Transcript(text="", segments=SEGS, words=WORDS + [
        WordSegment("let", 10.0, 10.2), WordSegment("me", 10.2, 10.3), WordSegment("again", 11.5, 12.0),
        WordSegment("The", 14.0, 14.2), WordSegment("dashboard", 16.0, 17.0),
    ])
    chunks = build_chunks([(13.5, False, False)], t, 20.0)
    assert chunks[0].transcript.endswith("let me again")
    assert chunks[1].transcript == "The dashboard"
    # Without word timings, segments go to the chunk holding their midpoint.
    t.words = None
    chunks = build_chunks([(13.5, False, False)], t, 20.0)
    assert "dashboard" not in chunks[0].transcript and "dashboard" in chunks[1].transcript


def test_basic_retake_rule_keeps_the_latest_take():
    from video_chunker.splitter import compute_clean_segments
    segs = [
        TranscriptSegment("Hi, this is the intro", 0.0, 3.0),
        TranscriptSegment("So the first thing we, um, the first", 6.5, 10.5),
        TranscriptSegment("So the first thing we do is open the dashboard", 13.6, 17.0),
    ]
    keep = compute_clean_segments(Transcript(text="", segments=segs), [], 17.0)
    assert [(k.start, k.end) for k in keep] == [(0.0, 6.5), (13.6, 17.0)]


def test_count_cuts_includes_trailing_and_leading():
    from video_chunker.splitter import CleanSegment, _count_cuts
    assert _count_cuts([CleanSegment(0, 10, "speech")], 17.0) == 1
    assert _count_cuts([CleanSegment(1, 5, "s"), CleanSegment(6, 17, "s")], 17.0) == 2
    assert _count_cuts([CleanSegment(0, 17, "kept")], 17.0) == 0
