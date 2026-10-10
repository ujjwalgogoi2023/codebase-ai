"""A/B test for follow-up questions: QUERY_REWRITER=concat vs llm.

Usage:
    python evaluation/followup_ab.py evaluation/answers_codebaseai.json

Uses the "followups" section of an answer_quality.py question set:

    "followups": [
      {"first": "How are citations validated?",
       "followup": "What happens to one that points at a missing source?",
       "facts": [["removed", "stripped"]]}
    ]

Each conversation's opening answer is generated ONCE and shared, so both modes
replay identical history and differ only in how the follow-up is handled.
Runs in-process against the database (not over HTTP), because the two modes
are a server setting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.db.base import session_scope  # noqa: E402
from app.models import Repository  # noqa: E402
from app.services.generator import Generator  # noqa: E402
from sqlalchemy import select  # noqa: E402

MODES = ("concat", "llm")


def find_repository(session, name: str) -> int:
    repository_id = session.scalars(
        select(Repository.id)
        .where(Repository.name == name, Repository.index_status == "ready")
        .order_by(Repository.id.desc())
    ).first()
    if repository_id is None:
        raise SystemExit(f"No ready repository named {name!r}. Index it first.")
    return repository_id


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare follow-up handling strategies.")
    ap.add_argument("question_set", type=Path)
    ap.add_argument("--repository-id", type=int, default=None)
    args = ap.parse_args()

    spec = json.loads(args.question_set.read_text(encoding="utf-8"))
    conversations = spec.get("followups", [])
    if not conversations:
        raise SystemExit("The question set has no 'followups' section.")

    base = get_settings()
    settings = {m: base.model_copy(update={"query_rewriter": m}) for m in MODES}
    score = dict.fromkeys(MODES, 0)

    with session_scope() as session:
        repository_id = args.repository_id or find_repository(session, spec["repository"])
        print(f"\n  {args.question_set.name} -> repository {repository_id}")

        for convo in conversations:
            opener = Generator(session, settings=settings["concat"]).answer(
                convo["first"], repository_id
            )
            history = [
                {"role": "user", "content": convo["first"]},
                {"role": "assistant", "content": opener.answer},
            ]
            print(f"\n  {convo['first']}\n    -> {convo['followup']}")

            for mode in MODES:
                started = time.time()
                result = Generator(session, settings=settings[mode]).answer(
                    convo["followup"], repository_id, history=history
                )
                text = result.answer.lower()
                ok = all(any(alt.lower() in text for alt in g) for g in convo["facts"])
                score[mode] += ok
                print(f"    {mode:6} {'OK  ' if ok else 'MISS'} {time.time() - started:5.1f}s"
                      f"  asked as: {result.search_query!r}")
                if not ok:
                    print(f"           answer: {' '.join(result.answer.split())[:180]!r}")

    n = len(conversations)
    print(f"\n  follow-ups answered correctly: "
          + ", ".join(f"{m} {score[m]}/{n}" for m in MODES) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
