"""Tests for the evaluation metrics (RAG pipeline stage 11).

Pure logic, no database and no model: given a ranked list and a set of expected
files, the score is arithmetic.

Worth testing carefully for an unusual reason. Everywhere else, a bug produces
a visibly wrong answer. Here, a bug produces a plausible NUMBER — and a number
is what ends up on a CV and in an interview. A metric that silently counts a
near-miss as a hit is worse than no metric at all.
"""

from __future__ import annotations

import json

import pytest

from app.services.evaluation import (
    EvalQuestion,
    EvalReport,
    first_hit_rank,
    first_symbol_hit_rank,
    format_report,
    load_dataset,
    normalize_path,
    score_question,
)


def question(**overrides) -> EvalQuestion:
    defaults = dict(
        id="q1",
        question="where is auth?",
        expected_files=["app/services/auth.py"],
        expected_symbols=[],
        category="auth",
    )
    defaults.update(overrides)
    return EvalQuestion(**defaults)


def report_with(*ranks: int | None, config: dict | None = None) -> EvalReport:
    """A report whose questions hit at the given ranks (None = miss)."""
    results = []
    for index, rank in enumerate(ranks):
        retrieved = ["other.py"] * 20
        if rank is not None:
            retrieved[rank - 1] = "app/services/auth.py"
        results.append(
            score_question(question(id=f"q{index}"), retrieved)
        )
    return EvalReport(
        config=config or {"embedding_provider": "openai"}, results=results
    )


# ---------------------------------------------------------------------------
# Hit ranking
# ---------------------------------------------------------------------------

def test_rank_is_one_based():
    assert first_hit_rank(["a.py", "b.py"], ["a.py"]) == 1
    assert first_hit_rank(["a.py", "b.py"], ["b.py"]) == 2


def test_missing_expected_file_returns_none():
    assert first_hit_rank(["a.py", "b.py"], ["c.py"]) is None


def test_first_of_several_expected_files_wins():
    """Any expected file counts; we want the earliest correct hit."""
    assert first_hit_rank(["x.py", "b.py", "a.py"], ["a.py", "b.py"]) == 2


def test_suffix_matching_allows_readable_labels():
    """A dataset saying 'services/chunker.py' must match the indexed path."""
    assert first_hit_rank(["app/services/chunker.py"], ["services/chunker.py"]) == 1
    assert first_hit_rank(["backend/app/main.py"], ["app/main.py"]) == 1


def test_suffix_matching_does_not_match_a_partial_filename():
    """'auth.py' must not be matched by 'oauth.py'."""
    assert first_hit_rank(["app/oauth.py"], ["auth.py"]) is None


def test_separator_and_case_differences_are_ignored():
    assert first_hit_rank(["App\\Services\\Auth.py"], ["app/services/auth.py"]) == 1


def test_empty_inputs():
    assert first_hit_rank([], ["a.py"]) is None
    assert first_hit_rank(["a.py"], []) is None


def test_symbol_matching_uses_the_last_segment():
    """'authenticate_user' should match 'AuthService.authenticate_user'."""
    symbols = [None, "Other.thing", "AuthService.authenticate_user"]
    assert first_symbol_hit_rank(symbols, ["authenticate_user"]) == 3
    assert first_symbol_hit_rank(symbols, ["missing_symbol"]) is None
    assert first_symbol_hit_rank(symbols, []) is None


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def test_top_k_accuracy_counts_hits_within_k():
    report = report_with(1, 2, 4, None)

    assert report.accuracy_at(1) == pytest.approx(0.25)
    assert report.accuracy_at(3) == pytest.approx(0.50)
    assert report.accuracy_at(5) == pytest.approx(0.75)


def test_top_k_accuracy_is_monotonic():
    """Top-5 can never be lower than Top-1. A drop means a broken metric."""
    report = report_with(1, 3, 5, 7, None, 2)
    values = [report.accuracy_at(k) for k in (1, 3, 5, 10)]
    assert values == sorted(values)


def test_mrr_rewards_earlier_hits():
    assert report_with(1).mrr == pytest.approx(1.0)
    assert report_with(2).mrr == pytest.approx(0.5)
    assert report_with(4).mrr == pytest.approx(0.25)
    assert report_with(None).mrr == pytest.approx(0.0)
    assert report_with(1, 2).mrr == pytest.approx(0.75)


def test_mrr_separates_configs_that_top_k_cannot():
    """Both score 0% Top-1 and 100% Top-5, but one ranks better."""
    better = report_with(2, 2, 2)
    worse = report_with(5, 5, 5)

    assert better.accuracy_at(1) == worse.accuracy_at(1)
    assert better.accuracy_at(5) == worse.accuracy_at(5)
    assert better.mrr > worse.mrr


def test_empty_report_does_not_divide_by_zero():
    empty = EvalReport(config={})
    assert empty.accuracy_at(3) == 0.0
    assert empty.mrr == 0.0
    assert empty.by_category() == {}


def test_misses_are_listed_for_debugging():
    report = report_with(1, None, 3, None)
    assert len(report.misses) == 2


def test_accuracy_by_category():
    results = [
        score_question(question(id="a", category="auth"), ["app/services/auth.py"]),
        score_question(question(id="b", category="auth"), ["nope.py"]),
        score_question(question(id="c", category="api"), ["app/services/auth.py"]),
    ]
    report = EvalReport(config={}, results=results)

    assert report.by_category() == {"api": 1.0, "auth": 0.5}


# ---------------------------------------------------------------------------
# Honesty about fake providers
# ---------------------------------------------------------------------------

def test_a_fake_run_is_marked_not_meaningful():
    """Random vectors produce numbers. They must not look like results."""
    fake = report_with(1, 1, config={"embedding_provider": "fake"})
    real = report_with(1, 1, config={"embedding_provider": "openai"})

    assert fake.is_meaningful is False
    assert real.is_meaningful is True


def test_the_warning_appears_in_the_printed_report():
    text = format_report(report_with(1, config={"embedding_provider": "fake"}))
    assert "NOT MEANINGFUL" in text
    assert "Do not quote these" in text


def test_no_warning_on_a_real_run():
    text = format_report(report_with(1, config={"embedding_provider": "openai"}))
    assert "NOT MEANINGFUL" not in text


def test_summary_carries_the_config_that_produced_it():
    """A number without its configuration is not a result."""
    summary = report_with(1, 2, config={"embedding_provider": "openai",
                                        "chunk_strategy": "naive"}).summary()

    assert summary["config"]["chunk_strategy"] == "naive"
    assert summary["questions"] == 2
    assert "top_3" in summary["accuracy"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def test_load_dataset(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(
        json.dumps(
            {
                "repository": "demo",
                "questions": [
                    {"id": "q1", "question": "where?", "expected_files": ["a.py"]},
                    {
                        "id": "q2",
                        "question": "how?",
                        "expected_files": ["b.py"],
                        "expected_symbols": ["go"],
                        "category": "api",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    name, questions = load_dataset(path)

    assert name == "demo"
    assert len(questions) == 2
    assert questions[1].category == "api"
    assert questions[1].expected_symbols == ["go"]


def test_a_single_expected_file_may_be_a_string(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(
        json.dumps({"questions": [{"id": "q1", "question": "?", "expected_files": "a.py"}]}),
        encoding="utf-8",
    )
    _, questions = load_dataset(path)
    assert questions[0].expected_files == ["a.py"]


def test_question_without_ground_truth_is_rejected():
    """An unlabelled question would silently count as a permanent miss."""
    with pytest.raises(ValueError, match="no expected_files"):
        EvalQuestion.from_dict({"id": "q1", "question": "where?"})


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {"id": "q1", "question": "a", "expected_files": ["a.py"]},
                    {"id": "q1", "question": "b", "expected_files": ["b.py"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate question ids"):
        load_dataset(path)


def test_normalize_path():
    assert normalize_path("./App\\X.py") == "app/x.py"


# ---------------------------------------------------------------------------
# The shipped dataset
# ---------------------------------------------------------------------------

def test_shipped_dataset_is_valid_and_large_enough():
    """Spec section 18 asks for 30-50 questions."""
    from pathlib import Path

    dataset = Path(__file__).resolve().parents[2] / "evaluation" / "dataset.json"
    name, questions = load_dataset(dataset)

    assert name
    assert len(questions) >= 30, "too small to be a meaningful evaluation set"
    assert all(q.expected_files for q in questions)
    assert len({q.id for q in questions}) == len(questions)
    # More than one category, or per-category reporting says nothing.
    assert len({q.category for q in questions}) >= 5


def test_shipped_dataset_points_at_files_that_exist():
    """A typo in a label makes a question permanently unanswerable."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    backend = root / "backend"
    _, questions = load_dataset(root / "evaluation" / "dataset.json")

    missing = [
        (q.id, expected)
        for q in questions
        for expected in q.expected_files
        if not (backend / expected).exists()
    ]
    assert not missing, f"dataset references files that do not exist: {missing}"


def test_a_fake_llm_makes_a_RERANKING_run_not_meaningful():
    """The fake LLM returns a constant, so reranking degenerates to identity.

    Scoring that as "reranking" would report a real-looking number for a
    comparison that never actually happened.
    """
    reranked = report_with(
        1,
        config={
            "embedding_provider": "openai",
            "reranker": "llm",
            "llm_provider": "fake",
        },
    )
    assert reranked.is_meaningful is False


def test_a_fake_llm_does_not_invalidate_a_retrieval_only_run():
    """Without reranking, the LLM plays no part in the ranking being measured."""
    retrieval_only = report_with(
        1,
        config={
            "embedding_provider": "openai",
            "reranker": "identity",
            "llm_provider": "fake",
        },
    )
    assert retrieval_only.is_meaningful is True
