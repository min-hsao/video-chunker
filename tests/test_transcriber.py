import json
from pathlib import Path

import pytest

from video_chunker import transcriber
from video_chunker.transcriber import parse_whisper_cpp_json, resolve_ggml_model, resolve_mode

SAMPLE = json.loads((Path(__file__).parent / "data" / "whisper-cpp-sample.json").read_text())


def test_parse_segments():
    t = parse_whisper_cpp_json(SAMPLE)
    assert [s.text for s in t.segments] == [
        "Hi, I'm Sam, and this is a quick demo of Samtrix.",
        "So the first thing we, UM, the first, let me try that again.",
        "The first thing we do is open the dashboard and pick a project.",
    ]
    assert t.segments[1].start == pytest.approx(6.8)
    assert t.text.startswith("Hi, I'm Sam")


def test_parse_words_join_subwords_and_punctuation():
    t = parse_whisper_cpp_json(SAMPLE)
    words = [w.word for w in t.words]
    assert words[:4] == ["Hi,", "I'm", "Sam,", "and"]
    assert "Samtrix." in words  # sub-word tokens joined
    assert not any(w.startswith("[_") for w in words)
    # Times are ordered and inside their segment; confidences are probabilities.
    for w in t.words:
        assert w.start <= w.end
        assert w.confidence is None or 0 <= w.confidence <= 1


def test_punctuation_does_not_stretch_word_end():
    data = {"transcription": [{
        "offsets": {"from": 0, "to": 2000}, "text": " Hi, there",
        "tokens": [
            {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}, "p": 0.5},
            {"text": " Hi", "offsets": {"from": 50, "to": 320}, "p": 0.9},
            {"text": ",", "offsets": {"from": 320, "to": 900}, "p": 0.9},
            {"text": " there", "offsets": {"from": 900, "to": 1200}, "p": 0.8},
        ],
    }]}
    t = parse_whisper_cpp_json(data)
    assert [(w.word, w.start, w.end) for w in t.words] == [("Hi,", 0.05, 0.32), ("there", 0.9, 1.2)]


def test_empty_segments_skipped():
    t = parse_whisper_cpp_json({"transcription": [{"offsets": {"from": 0, "to": 10}, "text": "  ", "tokens": []}]})
    assert t.segments == [] and t.words is None


def test_resolve_mode(monkeypatch):
    monkeypatch.setattr(transcriber.shutil, "which", lambda name: "/usr/bin/whisper-cli")
    assert resolve_mode("auto") == "cpp"
    monkeypatch.setattr(transcriber.shutil, "which", lambda name: None)
    assert resolve_mode("auto") == "local"
    assert resolve_mode("OpenAI") == "openai"


def test_resolve_ggml_model(tmp_path, monkeypatch):
    model = tmp_path / "ggml-custom.bin"
    model.write_bytes(b"x")
    assert resolve_ggml_model(str(model)) == model
    with pytest.raises(RuntimeError):
        resolve_ggml_model(str(tmp_path / "missing.bin"))
    monkeypatch.setattr(transcriber, "MODEL_DIR", tmp_path)
    (tmp_path / "ggml-small.en.bin").write_bytes(b"x")
    assert resolve_ggml_model("small.en") == tmp_path / "ggml-small.en.bin"
    with pytest.raises(RuntimeError):
        resolve_ggml_model("large-v3-turbo", download=False)


def test_timing_drift():
    from video_chunker.transcriber import Transcript, TranscriptSegment, WordSegment, timing_drift
    from video_chunker.utils import SilenceInterval
    silences = [SilenceInterval(3.2, 6.5)]
    good = Transcript("", [TranscriptSegment("a", 0, 3.2)], [WordSegment("a", 0.1, 3.1), WordSegment("b", 6.6, 9.0)])
    assert timing_drift(good, silences, 17.0) == 0
    drifted = Transcript("", [], [WordSegment("a", 0.1, 3.0), WordSegment("b", 3.0, 6.0)])
    assert timing_drift(drifted, silences, 17.0) > 0.4
    past_end = Transcript("", [TranscriptSegment("x", 10.0, 20.0)], None)
    assert timing_drift(past_end, [], 17.0) == pytest.approx(0.3)


def test_dtw_preset():
    from video_chunker.transcriber import dtw_preset
    assert dtw_preset(Path("ggml-large-v3-turbo.bin")) == "large.v3.turbo"
    assert dtw_preset(Path("ggml-large-v3-turbo-q5_0.bin")) == "large.v3.turbo"
    assert dtw_preset(Path("/x/ggml-base.en.bin")) == "base.en"
    assert dtw_preset(Path("my-finetune.bin")) is None


def test_dtw_times_override_stretched_offsets():
    def tok(text, a, b, dtw):
        return {"text": text, "offsets": {"from": a, "to": b}, "p": 0.9, "t_dtw": dtw}
    data = {"transcription": [
        {"offsets": {"from": 0, "to": 6480}, "text": " Hi Samtrix.",
         "tokens": [tok("[_BEG_]", 0, 0, 0), tok(" Hi", 100, 400, 14), tok(" Samt", 400, 4000, 60),
                    tok("rix", 4000, 5950, 70), tok(".", 5950, 6480, 290)]},
        {"offsets": {"from": 6480, "to": 9000}, "text": " So",
         "tokens": [tok(" So", 6530, 6760, 654)]},
    ]}
    t = parse_whisper_cpp_json(data)
    assert [w.word for w in t.words] == ["Hi", "Samtrix.", "So"]
    assert t.words[1].start == pytest.approx(0.60)
    assert t.words[1].end == pytest.approx(0.60 + 0.075 * 7)  # not stretched to 5.95 s
    assert t.words[2].start == pytest.approx(6.54)
