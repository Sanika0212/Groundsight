"""Sentence segmentation with exact character offsets.

Answers are checked claim by claim, and the report highlights claims and evidence in
place, so every segment keeps its ``start``/``end`` offsets into the original text
(``text[start:end] == segment.text``).

The splitter is rule-based and dependency-free. It splits on ``.``, ``!`` and ``?``
followed by whitespace, and on line breaks, but not after common abbreviations
("Dr.", "e.g."), initials ("J. Smith"), or before a lowercase continuation. List
markers and Markdown headings are not claims and are left out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof sr jr st mt ft vs etc inc ltd co corp llc plc no nos fig figs eq
    approx dept est univ gen gov sen rep rev hon capt lt col sgt cf al ca vol pp ed eds
    e.g i.e a.m p.m u.s u.k u.n ph.d m.d b.a m.a jan feb mar apr jun jul aug sep sept
    oct nov dec mon tue wed thu fri sat sun
    """.split()
)

_TERMINAL = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")
# "a." / "b)" / "A)" are list markers; "A." is not, since it is usually an initial.
_LIST_MARKER = re.compile(r"^\s*(?:[-*•‣▪]|\d{1,3}[.)]|[a-z][.)]|[A-Z]\))\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s")
_WORD = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class Segment:
    text: str
    start: int
    end: int


def _is_abbreviation(text: str, dot: int) -> bool:
    """Whether the ``.`` at ``dot`` ends an abbreviation or an initial."""
    start = dot
    while start > 0 and not text[start - 1].isspace() and text[start - 1] not in "([\"'“‘":
        start -= 1
    word = text[start:dot]
    if len(word) == 1 and word.isalpha() and word.isupper():
        return True  # an initial, as in "J. Smith"
    return word.lower() in ABBREVIATIONS


def _segment_line(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence boundaries inside ``text[start:end]`` (one line or soft-wrapped paragraph)."""
    spans = []
    cursor = start
    for match in _TERMINAL.finditer(text, start, end):
        punct_end = match.end()
        nxt = punct_end
        while nxt < end and text[nxt].isspace():
            nxt += 1
        if nxt >= end:
            break
        if text[nxt].islower() or text[nxt] in ",;:":
            continue  # "e.g. the", "approx. 5" style continuations
        single_dot = match.group(0).rstrip("\"'”’)]") == "."
        if single_dot and _is_abbreviation(text, match.start()):
            continue
        spans.append((cursor, punct_end))
        cursor = nxt
    spans.append((cursor, end))
    return spans


def _lines(text: str) -> list[tuple[int, int]]:
    """Logical lines: a line break ends a line unless the next line continues lowercase."""
    out: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n", text):
        rest = text[match.end() :]
        stripped = rest.lstrip(" \t")
        continues = (
            bool(stripped)
            and stripped[0].islower()
            and not _LIST_MARKER.match(rest)
            and text[start : match.start()].strip() != ""
        )
        if not continues:
            out.append((start, match.start()))
            start = match.end()
    out.append((start, len(text)))
    return out


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def sentences(text: str, *, min_words: int = 1) -> list[Segment]:
    """Split ``text`` into sentences with exact offsets.

    Segments without any letter or digit, list markers and Markdown headings are
    dropped; so are segments with fewer than ``min_words`` words.
    """
    segments = []
    for line_start, line_end in _lines(text):
        line = text[line_start:line_end]
        if _HEADING.match(line):
            continue
        marker = _LIST_MARKER.match(line)
        content_start = line_start + (marker.end() if marker else 0)
        for s, e in _segment_line(text, content_start, line_end):
            s, e = _trim(text, s, e)
            piece = text[s:e]
            if not _WORD.search(piece) or len(piece.split()) < min_words:
                continue
            segments.append(Segment(piece, s, e))
    return segments


def windows(text: str, *, max_words: int = 250) -> list[Segment]:
    """Cover ``text`` with windows of whole sentences of at most ``max_words`` words.

    Consecutive windows overlap by one sentence, so evidence that straddles a window
    boundary is still seen whole. A single sentence longer than the budget becomes
    its own window.
    """
    sents = sentences(text)
    if not sents:
        stripped_start, stripped_end = _trim(text, 0, len(text))
        if stripped_start == stripped_end:
            return []
        return [Segment(text[stripped_start:stripped_end], stripped_start, stripped_end)]
    out: list[Segment] = []
    i = 0
    while i < len(sents):
        words = 0
        j = i
        while j < len(sents) and (j == i or words + len(sents[j].text.split()) <= max_words):
            words += len(sents[j].text.split())
            j += 1
        start, end = sents[i].start, sents[j - 1].end
        out.append(Segment(text[start:end], start, end))
        if j >= len(sents):
            break
        i = max(j - 1, i + 1)
    return out
