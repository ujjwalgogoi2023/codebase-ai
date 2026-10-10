"""Retrieval evaluation: measuring whether the system actually works.

Stage 11, and the answer to the interview question that matters most:

    "How did you know your RAG system was working?"

The weak answer is "I used LangChain and Pinecone and it seemed good". The
strong answer is a labelled question set, a metric, and a number produced by
running it.

WHAT IS MEASURED, AND WHY IT IS RETRIEVAL
------------------------------------------
The metrics here score RETRIEVAL, not answers. That is deliberate:

1. Retrieval is the ceiling. If the chunk containing the answer never reaches
   the prompt, no model can answer correctly. Generation quality is bounded by
   retrieval quality, so retrieval is the thing worth optimising first.
2. Retrieval has cheap ground truth. "Which file answers this?" is a fact a
   developer can label in seconds and a script can check exactly. "Is this
   paragraph a good explanation?" needs a human or an LLM judge, and both are
   slow, expensive and noisy.
3. Retrieval metrics are stable. The same query over the same index gives the
   same result every time, so a difference between two runs is a real
   difference in the system, not sampling noise.

THE METRICS
-----------
Top-K accuracy: in what fraction of questions did an expected file appear in
the top K retrieved chunks? Reported at K = 1, 3, 5 because they answer
different questions. Top-1 is "did it nail it"; Top-5 is "was the answer
somewhere in what the LLM got to read". Top-3 matters most here, because
RERANK_TOP_K is 3 — Top-3 accuracy IS the ceiling on how often the LLM can
possibly answer correctly.

MRR (mean reciprocal rank): 1/rank of the first correct hit, averaged. Unlike
Top-K it is sensitive to WHERE in the list the hit landed, so it can tell a
configuration that moved the answer from rank 4 to rank 2 from one that did
nothing, even though both score identically on Top-1 and Top-5.

A NOTE ON HONESTY
-----------------
`EvalReport` records the providers it ran with, and `is_meaningful` is False
when embeddings came from the fake provider. Fake embeddings are random
vectors, so every number computed from them is noise. Reporting such a run as
if it measured something would be worse than not measuring at all, so the
report carries the disclaimer with it rather than relying on whoever reads it
to remember.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class EvalQuestion:
    """One labelled question.

    `expected_files` is a list because more than one file can legitimately
    answer a question ("where is auth handled?" may be both the route and the
    middleware). A hit on ANY expected file counts, which is the honest rule:
    the question is whether retrieval found a correct answer, not whether it
    found every one.
    """

    id: str
    question: str
    expected_files: list[str]
    expected_symbols: list[str] = field(default_factory=list)
    category: str = "general"
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> EvalQuestion:
        expected = data.get("expected_files") or []
        if isinstance(expected, str):
            expected = [expected]
        if not expected:
            raise ValueError(f"Question {data.get('id')!r} has no expected_files")
        return cls(
            id=str(data["id"]),
            question=data["question"],
            expected_files=list(expected),
            expected_symbols=list(data.get("expected_symbols") or []),
            category=data.get("category", "general"),
            note=data.get("note", ""),
        )


@dataclass
class QuestionResult:
    """What retrieval returned for one question."""

    question: EvalQuestion
    retrieved_files: list[str]
    retrieved_symbols: list[str | None] = field(default_factory=list)
    # 1-based rank of the first correct file, or None if never retrieved.
    hit_rank: int | None = None
    symbol_hit_rank: int | None = None

    def hit_at(self, k: int) -> bool:
        return self.hit_rank is not None and self.hit_rank <= k

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.hit_rank if self.hit_rank else 0.0

    def as_dict(self) -> dict:
        return {
            "id": self.question.id,
            "question": self.question.question,
            "category": self.question.category,
            "expected_files": self.question.expected_files,
            "hit_rank": self.hit_rank,
            "symbol_hit_rank": self.symbol_hit_rank,
            "retrieved_files": self.retrieved_files[:10],
        }


@dataclass
class EvalReport:
    """Scores for one configuration, plus what produced them."""

    config: dict
    results: list[QuestionResult] = field(default_factory=list)
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def total(self) -> int:
        return len(self.results)

    def accuracy_at(self, k: int) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.hit_at(k)) / len(self.results)

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    @property
    def symbol_accuracy_at_3(self) -> float:
        """Stricter: did we retrieve the right FUNCTION, not just the file?"""
        if not self.results:
            return 0.0
        hits = sum(
            1
            for r in self.results
            if r.symbol_hit_rank is not None and r.symbol_hit_rank <= 3
        )
        return hits / len(self.results)

    @property
    def misses(self) -> list[QuestionResult]:
        """Questions where no expected file was retrieved at all.

        The most useful output of an evaluation run. An aggregate score tells
        you the system is 72% right; the miss list tells you WHICH questions
        fail, which is where the next improvement comes from.
        """
        return [r for r in self.results if r.hit_rank is None]

    @property
    def is_meaningful(self) -> bool:
        """False when a fake provider was involved in producing the ranking.

        Fake embeddings are random vectors, so every number derived from them
        is noise. The fake LLM matters too: it returns a constant string, so a
        reranking run against it degenerates into the identity ranking and
        scores identically to no reranking at all. Reporting either as a result
        would be worse than not measuring.
        """
        if "fake" in str(self.config.get("embedding_provider", "")).lower():
            return False
        reranker = str(self.config.get("reranker", "")).lower()
        if reranker not in ("", "identity", "none", "off"):
            return "fake" not in str(self.config.get("llm_provider", "")).lower()
        return True

    def by_category(self) -> dict[str, float]:
        """Top-3 accuracy per category, to show WHERE it is weak."""
        buckets: dict[str, list[QuestionResult]] = {}
        for result in self.results:
            buckets.setdefault(result.question.category, []).append(result)
        return {
            category: sum(1 for r in items if r.hit_at(3)) / len(items)
            for category, items in sorted(buckets.items())
        }

    def summary(self) -> dict:
        return {
            "config": self.config,
            "created_at": self.created_at,
            "questions": self.total,
            "is_meaningful": self.is_meaningful,
            "accuracy": {f"top_{k}": round(self.accuracy_at(k), 4) for k in self.k_values},
            "mrr": round(self.mrr, 4),
            "symbol_top_3": round(self.symbol_accuracy_at_3, 4),
            "by_category": {k: round(v, 4) for k, v in self.by_category().items()},
            "misses": [r.question.id for r in self.misses],
        }

    def as_dict(self) -> dict:
        return {**self.summary(), "results": [r.as_dict() for r in self.results]}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_path(path: str) -> str:
    """Compare paths without tripping over separators or a leading './'."""
    return path.replace("\\", "/").strip().lstrip("./").lower()


def first_hit_rank(retrieved: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first retrieved item matching any expected value.

    Matching is suffix-based so a dataset can say "services/chunker.py" and
    still match "app/services/chunker.py". That keeps the labels readable
    without making them brittle to where the repository root happens to be.
    """
    wanted = [normalize_path(e) for e in expected if e]
    if not wanted:
        return None

    for index, item in enumerate(retrieved, start=1):
        if item is None:
            continue
        candidate = normalize_path(item)
        for target in wanted:
            if candidate == target or candidate.endswith("/" + target):
                return index
    return None


def first_symbol_hit_rank(
    retrieved_symbols: list[str | None], expected_symbols: list[str]
) -> int | None:
    """Rank of the first retrieved chunk whose symbol matches.

    Symbols are matched on the last dotted segment, so a dataset entry of
    "authenticate_user" matches a chunk labelled "AuthService.authenticate_user".
    """
    if not expected_symbols:
        return None

    wanted = {s.strip().lower().rsplit(".", 1)[-1] for s in expected_symbols if s}

    for index, symbol in enumerate(retrieved_symbols, start=1):
        if not symbol:
            continue
        if symbol.strip().lower().rsplit(".", 1)[-1] in wanted:
            return index
    return None


def score_question(
    question: EvalQuestion,
    retrieved_files: list[str],
    retrieved_symbols: list[str | None] | None = None,
) -> QuestionResult:
    retrieved_symbols = retrieved_symbols or []
    return QuestionResult(
        question=question,
        retrieved_files=retrieved_files,
        retrieved_symbols=retrieved_symbols,
        hit_rank=first_hit_rank(retrieved_files, question.expected_files),
        symbol_hit_rank=first_symbol_hit_rank(
            retrieved_symbols, question.expected_symbols
        ),
    )


def load_dataset(path: str | Path) -> tuple[str, list[EvalQuestion]]:
    """Load a dataset file, returning (repository_name, questions)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = [EvalQuestion.from_dict(q) for q in data["questions"]]

    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate question ids in dataset: {sorted(duplicates)}")

    return data.get("repository", ""), questions


def format_report(report: EvalReport) -> str:
    """Human-readable summary for the terminal."""
    lines: list[str] = []
    config = report.config

    lines.append("")
    lines.append(f"  Questions        {report.total}")
    lines.append(
        f"  Configuration    chunking={config.get('chunk_strategy')} "
        f"reranker={config.get('reranker')} top_k={config.get('top_k')}"
    )
    lines.append(
        f"  Providers        embeddings={config.get('embedding_provider')} "
        f"({config.get('embedding_model')})"
    )
    lines.append("  " + "-" * 58)

    for k in report.k_values:
        accuracy = report.accuracy_at(k)
        bar = "#" * round(accuracy * 30)
        lines.append(f"  Top-{k:<2} accuracy    {accuracy:6.1%}  {bar}")

    lines.append(f"  MRR              {report.mrr:6.3f}")
    if any(r.question.expected_symbols for r in report.results):
        lines.append(f"  Symbol Top-3     {report.symbol_accuracy_at_3:6.1%}")

    categories = report.by_category()
    if len(categories) > 1:
        lines.append("  " + "-" * 58)
        lines.append("  Top-3 by category")
        for category, accuracy in categories.items():
            lines.append(f"    {category:<22} {accuracy:6.1%}")

    if report.misses:
        lines.append("  " + "-" * 58)
        lines.append(f"  Missed entirely ({len(report.misses)}):")
        for result in report.misses[:10]:
            lines.append(f"    {result.question.id}  {result.question.question[:52]}")
        if len(report.misses) > 10:
            lines.append(f"    ... and {len(report.misses) - 10} more")

    if not report.is_meaningful:
        lines.append("  " + "=" * 58)
        lines.append("  THESE NUMBERS ARE NOT MEANINGFUL.")
        lines.append("  The run used the fake embedding provider, which returns")
        lines.append("  random vectors. Set EMBEDDING_PROVIDER=ollama (or openai) and")
        lines.append("  re-run to measure anything real. Do not quote these.")
        lines.append("  " + "=" * 58)

    lines.append("")
    return "\n".join(lines)
