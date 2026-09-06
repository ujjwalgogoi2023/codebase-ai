"""Run the evaluation for a single configuration.

Usage:
    python evaluation/evaluate_retrieval.py
    python evaluation/evaluate_retrieval.py --strategy naive --reranker llm
    python evaluation/evaluate_retrieval.py --top-k 5 --save results/run.json

Answers "how did you know your RAG system was working?" with a number instead
of an opinion.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import PROJECT_ROOT, RunConfig, load_questions, prepare, run_config  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.config import get_settings  # noqa: E402
from app.services.evaluation import format_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure retrieval accuracy.")
    ap.add_argument("--dataset", default=str(Path(__file__).parent / "dataset.json"))
    ap.add_argument("--strategy", default="code_aware", choices=["code_aware", "naive"])
    ap.add_argument("--reranker", default="identity", choices=["identity", "llm"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-chunk-lines", type=int, default=120)
    ap.add_argument("--save", help="write the full report as JSON")
    args = ap.parse_args()

    settings = get_settings()
    prepare(settings)

    base_name, source_path, questions = load_questions(Path(args.dataset))
    print(f"\n  Dataset      {Path(args.dataset).name}  ({len(questions)} questions)")
    print(f"  Repository   {source_path}")

    config = RunConfig(
        label=f"{args.strategy}+{args.reranker}",
        chunk_strategy=args.strategy,
        reranker=args.reranker,
        top_k=args.top_k,
        max_chunk_lines=args.max_chunk_lines,
    )

    report = run_config(config, questions, base_name, source_path, settings)
    print(format_report(report))

    if args.save:
        path = Path(args.save)
        if not path.is_absolute():
            path = Path(__file__).parent / path
        print(f"  saved to {report.save(path)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
