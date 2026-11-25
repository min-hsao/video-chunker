from video_chunker.grounding import find_segment_with_quote, normalise, quote_matches
from video_chunker.transcriber import TranscriptSegment


def test_normalise():
    assert normalise("Hello, World!  ") == "helloworld"
    assert normalise("ＡＢＣ１２") == "abc12"  # NFKC full-width


def test_quote_matches_ignores_case_and_punctuation():
    assert quote_matches("so the first thing we", "So the first thing we, UM, the first")
    assert quote_matches("UM", "we, um, the")
    assert not quote_matches("open the dashboard", "So the first thing we")


def test_quote_ellipsis_skips_words_in_order():
    text = "So the first thing we, um, the first, let me try that again"
    assert quote_matches("so the first ... try that again", text)
    assert quote_matches("so the first … again", text)
    assert not quote_matches("try that again ... so the first", text)


def test_empty_quote_never_matches():
    assert not quote_matches("", "anything")
    assert not quote_matches("...", "anything")


def test_find_segment_with_quote():
    segs = [TranscriptSegment("intro words here", 0, 1), TranscriptSegment("the dashboard is open", 1, 2)]
    assert find_segment_with_quote("the dashboard is open", segs) == 1
    assert find_segment_with_quote("um", segs) is None  # too short to re-point
    assert find_segment_with_quote("nothing like this at all", segs) is None
