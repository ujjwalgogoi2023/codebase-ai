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

        if result.sources:
            rule("SOURCES")
            for index, source in enumerate(result.sources, start=1):
                label = f" ({source.symbol})" if source.symbol else ""
                print(f"  [{index}] {source.citation}{label}   score {source.score:.4f}")
        if result.sources_omitted:
            print(f"\n  ({result.sources_omitted} weaker sources dropped to fit the context budget)")

        rule()
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
