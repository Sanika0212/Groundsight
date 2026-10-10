from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from groundsight.text import sentences, windows


def texts(value: str) -> list[str]:
    return [s.text for s in sentences(value)]


def test_splits_on_terminal_punctuation_and_keeps_offsets() -> None:
    text = "Dr. Lee said the dose is 2.5 mg. Is it safe? Yes!\nTake it daily."
    segs = sentences(text)
    assert [s.text for s in segs] == [
        "Dr. Lee said the dose is 2.5 mg.",
        "Is it safe?",
        "Yes!",
        "Take it daily.",
    ]
    assert all(text[s.start : s.end] == s.text for s in segs)


def test_abbreviations_initials_and_lowercase_continuations() -> None:
    assert texts("Prices rose e.g. in Q3. Then they fell.") == [
        "Prices rose e.g. in Q3.",
        "Then they fell.",
    ]
    assert texts("J. R. R. Tolkien wrote it. It sold well.") == [
        "J. R. R. Tolkien wrote it.",
        "It sold well.",
    ]
    assert texts("It costs approx. 5 dollars. That is cheap.") == [
        "It costs approx. 5 dollars.",
        "That is cheap.",
    ]
    assert texts("The U.S. Army left. Others stayed.") == ["The U.S. Army left.", "Others stayed."]
    assert texts('He said "stop." Then he left.') == ['He said "stop."', "Then he left."]


def test_lists_headings_and_soft_wraps() -> None:
    text = (
        "## Summary\n- First point here\n- Second point\n\n1. Numbered item.\n"
        "A paragraph that\nwraps onto a new line."
    )
    assert texts(text) == [
        "First point here",
        "Second point",
        "Numbered item.",
        "A paragraph that\nwraps onto a new line.",
    ]


def test_drops_empty_and_symbol_only_segments() -> None:
    assert texts("") == []
    assert texts("  \n\n ... \n - \n") == []
    assert texts("Ok.") == ["Ok."]
    assert [s.text for s in sentences("A b c. D.", min_words=2)] == ["A b c."]


def test_windows_cover_text_with_overlap() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(10))
    wins = windows(text, max_words=12)
    assert all(text[w.start : w.end] == w.text for w in wins)
    assert wins[0].start == 0
    assert wins[-1].end == len(text)
    for a, b in pairwise(wins):
        assert b.start < a.end  # one sentence of overlap
    assert windows("   ") == []
    assert [w.text for w in windows("no punctuation at all")] == ["no punctuation at all"]
    long = "word " * 400 + "end."
    assert len(windows(long, max_words=50)) == 1


@given(st.text(alphabet=st.sampled_from(list("ab .!?\nD-1.,'\"")), max_size=80))
def test_offsets_are_exact_ordered_and_disjoint(text: str) -> None:
    segs = sentences(text)
    for s in segs:
        assert text[s.start : s.end] == s.text
        assert s.text == s.text.strip()
    for a, b in pairwise(segs):
        assert a.end <= b.start
    for w in windows(text, max_words=5):
        assert text[w.start : w.end] == w.text
