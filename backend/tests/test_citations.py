"""Tests for citation resolution (RAG pipeline stage 7).

These need no database and no model: given an answer string and the sources
that were sent, the mapping back to file and line ranges is pure logic.

The cases that matter most are the adversarial ones — a fabricated [7], and an
answer containing real code with real brackets in it.
"""

from __future__ import annotations

import pytest

from app.services.citations import (
    Citation,
    attach_citations,
    citation_coverage,
    find_markers,
    format_sources_block,
    mask_code,
)
from app.services.retriever import RetrievedChunk


def chunk(index: int = 1, **overrides) -> RetrievedChunk:
    defaults = dict(
        chunk_id=100 + index,
        file_path=f"backend/file{index}.py",
        language="python",
        symbol=f"func_{index}",
        kind="function",
        start_line=10 * index,
        end_line=10 * index + 5,
        content=f"def func_{index}(): ...",
        distance=0.1,
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


SOURCES = [chunk(1), chunk(2), chunk(3)]


# ---------------------------------------------------------------------------
# Valid markers
# ---------------------------------------------------------------------------

def test_single_marker_resolves_to_a_real_location():
    result = attach_citations("Auth happens in the login handler [1].", SOURCES)

    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.index == 1
    assert citation.file_path == "backend/file1.py"
    assert citation.citation == "backend/file1.py:10-15"
    assert citation.symbol == "func_1"
    assert result.invalid_markers == []


def test_multiple_and_adjacent_markers():
    result = attach_citations("Tokens are checked [1][3] before routing [2].", SOURCES)

    assert [c.index for c in result.citations] == [1, 2, 3]


def test_comma_separated_marker():
    result = attach_citations("Both paths apply [1, 3].", SOURCES)
    assert [c.index for c in result.citations] == [1, 3]


def test_range_marker():
    result = attach_citations("See [1-3] for the full flow.", SOURCES)
    assert [c.index for c in result.citations] == [1, 2, 3]


def test_repeated_marker_is_reported_once():
    result = attach_citations("First [1]. Second [1]. Third [1].", SOURCES)
    assert len(result.citations) == 1


def test_citations_are_ordered_by_index():
    result = attach_citations("Later [3] then earlier [1].", SOURCES)
    assert [c.index for c in result.citations] == [1, 3]


def test_unused_sources_are_reported():
    """Useful signal: retrieval sent code the answer never needed."""
    result = attach_citations("Only the first matters [1].", SOURCES)

    assert [c.index for c in result.citations] == [1]
    assert [c.index for c in result.unused_sources] == [2, 3]


# ---------------------------------------------------------------------------
# Fabricated markers
# ---------------------------------------------------------------------------

def test_out_of_range_marker_is_caught_and_removed():
    """The model cited a seventh source when only three were sent."""
    result = attach_citations("Stripe is handled in payments [7].", SOURCES)

    assert result.invalid_markers == [7]
    assert result.citations == []
    assert "[7]" not in result.answer
    assert result.is_grounded is False


def test_raw_answer_is_preserved_for_auditing():
    result = attach_citations("Claim [9].", SOURCES)

    assert "[9]" in result.raw_answer, "the model's real output must be recoverable"
    assert "[9]" not in result.answer


def test_zero_is_not_a_valid_citation():
    """Sources are numbered from 1; [0] is either a mistake or an array index."""
    result = attach_citations("Something [0].", SOURCES)
    assert result.invalid_markers == [0]


def test_mixed_marker_keeps_its_valid_half():
    result = attach_citations("Partly right [2,9].", SOURCES)

    assert [c.index for c in result.citations] == [2]
    assert result.invalid_markers == [9]
    assert "[2,9]" in result.answer, "a marker with a valid part must not be removed"


def test_whitespace_is_tidied_after_removal():
    result = attach_citations("The token is checked [9] before routing.", SOURCES)

    assert "  " not in result.answer
    assert result.answer == "The token is checked before routing."


def test_marker_before_punctuation_leaves_no_gap():
    result = attach_citations("It is validated [8].", SOURCES)
    assert result.answer == "It is validated."


def test_strip_invalid_can_be_disabled():
    result = attach_citations("Claim [9].", SOURCES, strip_invalid=False)

    assert "[9]" in result.answer
    assert result.invalid_markers == [9]


def test_absurd_range_is_not_expanded():
    """[1-9999] must not allocate ten thousand invalid markers."""
    result = attach_citations("See [1-9999].", SOURCES)
    assert len(result.invalid_markers) < 50


# ---------------------------------------------------------------------------
# The bracket trap: answers contain code
# ---------------------------------------------------------------------------

def test_array_indexing_in_prose_is_not_a_citation():
    result = attach_citations("The handler reads items[0] and returns rows[2].", SOURCES)

    assert result.invalid_markers == [], "array index was parsed as a citation"
    assert result.citations == []
    assert "items[0]" in result.answer, "code was corrupted"
    assert "rows[2]" in result.answer


def test_call_then_index_is_not_a_citation():
    result = attach_citations("It calls get_users()[1] internally.", SOURCES)

    assert result.citations == []
    assert "get_users()[1]" in result.answer


def test_inline_code_span_is_ignored():
    result = attach_citations("Use `config[0]` for the default [1].", SOURCES)

    assert [c.index for c in result.citations] == [1]
    assert "`config[0]`" in result.answer


def test_fenced_code_block_is_ignored():
    answer = (
        "The parser indexes the list [1]:\n\n"
        "```python\n"
        "first = files[0]\n"
        "second = files[9]\n"
        "```\n"
    )
    result = attach_citations(answer, SOURCES)

    assert [c.index for c in result.citations] == [1]
    assert result.invalid_markers == []
    assert "files[9]" in result.answer, "code inside a fence was modified"


def test_mask_code_preserves_length_and_newlines():
    text = "a `b` c\n```\nxx\n```\nd"
    masked = mask_code(text)

    assert len(masked) == len(text)
    assert masked.count("\n") == text.count("\n")
    assert "b" not in masked and "xx" not in masked


# ---------------------------------------------------------------------------
# Coverage and grounding signals
# ---------------------------------------------------------------------------

def test_coverage_is_one_when_every_sentence_cites():
    text = "Auth lives in the handler [1]. Tokens are checked by middleware [2]."
    assert citation_coverage(text) == pytest.approx(1.0)


def test_coverage_is_partial_when_claims_go_uncited():
    text = (
        "Auth lives in the handler [1]. "
        "It probably also refreshes the session automatically. "
        "The cache is warmed on every startup of the service."
    )
    assert 0.0 < citation_coverage(text) < 1.0


def test_coverage_ignores_short_fragments():
    assert citation_coverage("Yes. No. Auth is in the handler [1].") == pytest.approx(1.0)


def test_coverage_of_uncited_answer_is_zero():
    assert citation_coverage("The payment flow is handled by the billing service.") == 0.0


def test_grounded_requires_citations_and_no_fabrications():
    assert attach_citations("Handled here [1].", SOURCES).is_grounded is True
    assert attach_citations("Handled here [9].", SOURCES).is_grounded is False
    assert attach_citations("Handled somewhere.", SOURCES).is_grounded is False
    assert attach_citations("Mostly [1] but also [9].", SOURCES).is_grounded is False


# ---------------------------------------------------------------------------
# Edge cases and rendering
# ---------------------------------------------------------------------------

def test_empty_answer():
    result = attach_citations("", SOURCES)
    assert result.answer == "" and result.citations == []


def test_no_sources_makes_every_marker_invalid():
    result = attach_citations("Claim [1].", [])
    assert result.invalid_markers == [1]
    assert result.citations == []


def test_source_without_a_symbol():
    result = attach_citations("Set up here [1].", [chunk(1, symbol=None, kind="module")])
    citation = result.citations[0]

    assert citation.symbol is None
    assert citation.label == citation.citation


def test_find_markers_reports_positions():
    markers = find_markers("a [1] b [2,3] c")
    assert [numbers for _, numbers in markers] == [[1], [2, 3]]


def test_sources_block_rendering():
    result = attach_citations("Auth [1] and orders [3].", SOURCES)
    block = format_sources_block(result.citations)

    assert block.startswith("Sources:")
    assert "- [1] backend/file1.py:10-15  (func_1)" in block
    assert "- [3] backend/file3.py:30-35  (func_3)" in block
    assert "file2.py" not in block, "uncited source listed as a source"


def test_sources_block_is_empty_without_citations():
    assert format_sources_block([]) == ""


def test_citation_serializes():
    payload = Citation.from_chunk(1, chunk(1)).as_dict()
    assert payload["citation"] == "backend/file1.py:10-15"
    assert payload["chunk_id"] == 101
    assert set(payload) >= {"index", "file_path", "start_line", "end_line", "score"}


def test_chained_index_expression_is_not_a_citation():
    """`grid[1][2]` must not become citations 1 and 2."""
    result = attach_citations("It walks grid[1][2] to find the cell.", SOURCES)

    assert result.citations == []
    assert result.invalid_markers == []
    assert "grid[1][2]" in result.answer


def test_adjacent_citations_and_chained_indexes_together():
    result = attach_citations(
        "Both handlers apply [1][2], reading matrix[0][1] each time.", SOURCES
    )

    assert [c.index for c in result.citations] == [1, 2]
    assert "matrix[0][1]" in result.answer
    assert result.invalid_markers == []
