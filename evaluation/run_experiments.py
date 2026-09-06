"""Sweep configurations and produce the comparison table (spec section 21).

Usage:
    python evaluation/run_experiments.py
    python evaluation/run_experiments.py --quick

Writes results/experiments.json and results/experiments.md.

The point of this file is that "code-aware chunking is better" and "reranking
helps" are HYPOTHESES until something measures them. Each row below changes
exactly one variable against the same corpus, the same questions and the same
scoring, so a difference in the numbers is attributable to that variable.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import PROJECT_ROOT, RunConfig, load_questions, prepare, run_config  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.config import get_settings  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

# Each entry isolates one variable against the row above it.
GRID = [
    RunConfig("naive chunking", chunk_strategy="naive", reranker="identity"),
    RunConfig("code-aware chunking", chunk_strategy="code_aware", reranker="identity"),
    RunConfig("code-aware + reranking", chunk_strategy="code_aware", reranker="llm"),
]

TOP_K_SWEEP = [
    RunConfig("code-aware, top_k=3", chunk_strategy="code_aware", top_k=3),
    RunConfig("code-aware, top_k=5", chunk_strategy="code_aware", top_k=5),
    RunConfig("code-aware, top_k=10", chunk_strategy="code_aware", top_k=10),
]

CHUNK_SIZE_SWEEP = [
    RunConfig("chunk_lines=40", chunk_strategy="code_aware", max_chunk_lines=40),
    RunConfig("chunk_lines=120", chunk_strategy="code_aware", max_chunk_lines=120),
    RunConfig("chunk_lines=250", chunk_strategy="code_aware", max_chunk_lines=250),
]


def markdown_table(reports) -> str:
    header = (
        "| Configuration | Top-1 | Top-3 | Top-5 | MRR |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    rows = "".join(
        f"| {r.config['label']} | {r.accuracy_at(1):.1%} | {r.accuracy_at(3):.1%} "
        f"| {r.accuracy_at(5):.1%} | {r.mrr:.3f} |\n"
        for r in reports
    )
    return header + rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the retrieval experiments.")
    ap.add_argument("--dataset", default=str(Path(__file__).parent / "dataset.json"))
    ap.add_argument("--quick", action="store_true", help="main grid only")
    args = ap.parse_args()

    settings = get_settings()
    prepare(settings)

    base_name, source_path, questions = load_questions(Path(args.dataset))
    grid = GRID if args.quick else GRID + TOP_K_SWEEP + CHUNK_SIZE_SWEEP

    print(f"\n  {len(questions)} questions x {len(grid)} configurations\n")

    reports = []
    for config in grid:
        report = run_config(config, questions, base_name, source_path, settings)
        reports.append(report)
        print(
            f"    -> Top-1 {report.accuracy_at(1):.1%}  "
            f"Top-3 {report.accuracy_at(3):.1%}  "
            f"Top-5 {report.accuracy_at(5):.1%}  MRR {report.mrr:.3f}\n"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "experiments.json").write_text(
        json.dumps([r.as_dict() for r in reports], indent=2), encoding="utf-8"
    )

    meaningful = all(r.is_meaningful for r in reports)
    table = markdown_table(reports)
    disclaimer = (
        ""
        if meaningful
        else (
            "\n> **These numbers are not meaningful.** This run used the fake "
            "embedding provider,\n> which returns random vectors. Re-run with "
            "`EMBEDDING_PROVIDER=openai` to measure anything real.\n"
        )
    )
    (RESULTS_DIR / "experiments.md").write_text(
        f"# Retrieval experiments\n\n"
        f"{len(questions)} questions, dataset `{Path(args.dataset).name}`.\n"
        f"{disclaimer}\n{table}",
        encoding="utf-8",
    )

    print(table)
    print(f"  written to {RESULTS_DIR}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
