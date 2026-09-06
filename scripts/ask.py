"""Dev CLI for pipeline stage 6 — ask a question about an indexed repository.

Usage:
    python scripts/ask.py 2 "where are passwords hashed?"
    python scripts/ask.py 2 "how does indexing work?" --show-prompt
    python scripts/ask.py 2 "what is the payment flow?" --top-k 5

--show-prompt prints the exact text the LLM receives. That is worth looking at
at least once: RAG is often described as if something clever happens at
generation time, and seeing the prompt makes clear that the cleverness all
happened earlier, in retrieval. The model just reads what it was handed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.db.base import session_scope  # noqa: E402
from app.services.embeddings import get_embedding_provider  # noqa: E402
from app.services.generator import (  # noqa: E402
    SYSTEM_PROMPT,
    Generator,
    build_user_message,
)
from app.services.llm import get_llm_provider  # noqa: E402
from app.services.retriever import RetrievalError, SearchFilters  # noqa: E402


def rule(title: str = "") -> None:
    print(f"\n  {'-' * 76}")
    if title:
        print(f"  {title}")
        print(f"  {'-' * 76}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Ask a question about a repository.")
    ap.add_argument("repository_id", type=int)
    ap.add_argument("question")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--language", action="append", dest="languages")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the exact prompt sent to the model")
    args = ap.parse_args()

    settings = get_settings()
    filters = SearchFilters(languages=args.languages)

    with session_scope() as session:
        generator = Generator(
            session,
            llm=get_llm_provider(settings),
            embedding_provider=get_embedding_provider(settings),
            settings=settings,
        )
        try:
            result = generator.answer(
                args.question,
                args.repository_id,
                top_k=args.top_k,
                filters=None if filters.is_empty() else filters,
            )
        except RetrievalError as exc:
            print(f"\n  error: {exc}\n")
            return 1

        if args.show_prompt:
            rule("SYSTEM PROMPT")
            for line in SYSTEM_PROMPT.splitlines():
                print(f"  {line}")
            rule("USER MESSAGE")
            message = build_user_message(args.question, result.sources)
            for line in message.splitlines():
                print(f"  {line}")

        rule("ANSWER")
        print()
        for line in result.answer.splitlines():
            print(f"  {line}")

        if result.citations:
            rule("SOURCES (cited by the answer)")
            for citation in result.citations:
                label = f"  ({citation.symbol})" if citation.symbol else ""
                print(f"  [{citation.index}] {citation.citation}{label}"
                      f"   score {citation.score:.4f}")

        if result.unused_sources:
            print(f"\n  retrieved but never cited: "
                  + ", ".join(c.citation for c in result.unused_sources))

        if result.invalid_markers:
            rule("HALLUCINATED CITATIONS CAUGHT")
            print(f"  the model cited {result.invalid_markers} but only "
                  f"{len(result.sources)} sources were provided")
            print("  these markers were removed from the answer above")
            print(f"  raw model output: {result.raw_answer!r}")

        if result.sources_omitted:
            print(f"\n  ({result.sources_omitted} weaker sources dropped to fit the context budget)")

        if result.rerank:
            r = result.rerank
            rule("RERANKING")
            print(f"  strategy {r.strategy}   {r.candidates} candidates -> "
                  f"{len(result.sources)} sent to the LLM")
            print(f"  changed the order: {r.changed_order}"
                  + ("   (model output unusable, fell back to vector order)"
                     if r.fell_back else ""))
            if r.llm_calls:
                print(f"  extra LLM calls: {r.llm_calls}"
                      + (f"   ~${r.usd}" if r.usd is not None else ""))

        rule()
        print(f"  grounded {result.is_grounded}   "
              f"citation coverage {result.citation_coverage:.0%}")
        print(f"  model {result.model}   provider {settings.llm_provider}")
        print(f"  tokens in {result.input_tokens}  out {result.output_tokens}"
              + (f"   ~${result.usd}" if result.usd is not None else ""))
        if result.no_context:
            print("  no_context: retrieval returned nothing, the model was never called")
        if settings.llm_provider == "fake":
            print("\n  NOTE: LLM_PROVIDER=fake -- the answer text above is canned.")
            print("        The prompt, retrieval and citations are real.")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
