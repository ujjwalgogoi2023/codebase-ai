"""Answer-quality test: are /chat answers correct, cited, and honest?

Usage:
    python evaluation/answer_quality.py evaluation/answers_codebaseai.json
    python evaluation/answer_quality.py my_set.json --repository-id 22

Retrieval accuracy (evaluate_retrieval.py) is the ceiling; this measures what
the user actually reads. It runs against the LIVE API, end to end, so start the
backend first and index the repository the question set describes.

A question set is JSON:

    {
      "repository": "codebaseai-backend",          # looked up by name
      "answerable": [
        {"question": "...",
         "expected_files": ["app/config.py"],       # matched by suffix
         "facts": [["10"], ["3", "three"]]}         # every group must appear;
      ],                                            # any one alternative will do
      "unanswerable": ["How does the app send email?"],
      "followups": [...]                            # used by followup_ab.py
    }

Facts are checked as case-insensitive substrings of the answer, so write them
as the specific values a correct answer cannot avoid ("300", "argon2"), not
as phrasing. Unanswerable questions must be about features verified absent
from the code: the right answer is a refusal.

Results are written NEXT TO the question set, as <name>.results.json, so a set
kept somewhere private (data/ is gitignored) keeps its answers private too.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

# Phrases that mean "the context does not show this". The system prompt asks
# for exactly this kind of plain statement rather than a guess.
REFUSAL_MARKERS = (
    "doesn't show", "does not show", "does not contain", "doesn't contain",
    "no information", "not shown", "not present", "isn't", "is not",
    "there is no", "couldn't find", "could not find", "not included",
    "no mention", "does not include", "doesn't include", "not provide",
    "does not provide", "doesn't provide", "no code",
)


def find_repository(client: httpx.Client, name: str) -> int:
    ready = [
        r for r in client.get("/repositories").json()
        if r["name"] == name and r["index_status"] == "ready"
    ]
    if not ready:
        raise SystemExit(
            f"No ready repository named {name!r}. Index it first, or pass "
            "--repository-id."
        )
    return max(r["id"] for r in ready)  # the most recent index


def ask(client: httpx.Client, repository_id: int, question: str) -> dict:
    started = time.time()
    response = client.post(
        "/chat", json={"repository_id": repository_id, "question": question}
    )
    response.raise_for_status()
    data = response.json()
    data["seconds"] = round(time.time() - started, 1)
    return data


def matches(path: str, expected: list[str]) -> bool:
    return any(path.endswith(e) for e in expected)


def is_refusal(answer: dict) -> bool:
    text = answer["answer"].lower()
    return answer["no_context"] or any(m in text for m in REFUSAL_MARKERS)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score /chat answers against a question set.")
    ap.add_argument("question_set", type=Path)
    ap.add_argument("--repository-id", type=int, default=None)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    spec = json.loads(args.question_set.read_text(encoding="utf-8"))
    client = httpx.Client(base_url=args.api, timeout=600)
    repository_id = args.repository_id or find_repository(client, spec["repository"])

    print(f"\n  {args.question_set.name} -> repository {repository_id}")
    answerable, unanswerable = [], []

    print(f"\n  ANSWERABLE ({len(spec['answerable'])})")
    for item in spec["answerable"]:
        a = ask(client, repository_id, item["question"])
        text = a["answer"].lower()
        missing = [g for g in item["facts"] if not any(alt.lower() in text for alt in g)]
        row = {
            "question": item["question"],
            "retrieved": any(matches(s["file_path"], item["expected_files"]) for s in a["sources"]),
            "cited": any(matches(c["file_path"], item["expected_files"]) for c in a["citations"]),
            "correct": not missing,
            "missing_facts": missing,
            "invalid_markers": a["invalid_markers"],
            "seconds": a["seconds"],
            "answer": a["answer"],
        }
        # A refusal when the facts WERE findable is the costly failure mode.
        row["wrongly_refused"] = not row["correct"] and is_refusal(a)
        answerable.append(row)

        flag = "OK  " if row["correct"] and row["cited"] else (
            "MISS" if not row["retrieved"] else "PART")
        print(f"  {flag} retrieved={int(row['retrieved'])} cited={int(row['cited'])} "
              f"facts={int(row['correct'])} {a['seconds']:5.1f}s  {item['question']}")
        if flag != "OK  ":
            print(f"         missing: {missing}  answer: {' '.join(a['answer'].split())[:180]!r}")

    print(f"\n  UNANSWERABLE ({len(spec.get('unanswerable', []))})")
    for question in spec.get("unanswerable", []):
        a = ask(client, repository_id, question)
        row = {
            "question": question,
            "refused": is_refusal(a),
            "invalid_markers": a["invalid_markers"],
            "seconds": a["seconds"],
            "answer": a["answer"],
        }
        unanswerable.append(row)
        print(f"  {'OK  ' if row['refused'] else 'FAIL'} {a['seconds']:5.1f}s  {question}")
        if not row["refused"]:
            print(f"         answer: {' '.join(a['answer'].split())[:180]!r}")

    n = len(answerable)
    summary = {
        "answerable": n,
        "retrieved": sum(r["retrieved"] for r in answerable),
        "cited": sum(r["cited"] for r in answerable),
        "correct": sum(r["correct"] for r in answerable),
        "wrongly_refused": sum(r["wrongly_refused"] for r in answerable),
        "unanswerable": len(unanswerable),
        "refused": sum(r["refused"] for r in unanswerable),
        "invalid_markers": sum(len(r["invalid_markers"]) for r in answerable + unanswerable),
    }
    times = sorted(r["seconds"] for r in answerable + unanswerable)

    print("\n  SUMMARY")
    print(f"  right file in the sources sent to the model : {summary['retrieved']}/{n}")
    print(f"  right file cited                            : {summary['cited']}/{n}")
    print(f"  key facts in the answer                     : {summary['correct']}/{n}")
    print(f"  said 'not shown' when it was there          : {summary['wrongly_refused']}/{n}")
    print(f"  refused a feature that does not exist       : "
          f"{summary['refused']}/{summary['unanswerable']}")
    print(f"  invalid citation markers                    : {summary['invalid_markers']}")
    if times:
        print(f"  seconds per question                        : "
              f"median {times[len(times) // 2]:.1f}, max {times[-1]:.1f}")

    out = args.question_set.with_name(f"{args.question_set.stem}.results.json")
    out.write_text(
        json.dumps(
            {"question_set": args.question_set.name, "repository_id": repository_id,
             "summary": summary, "answerable": answerable, "unanswerable": unanswerable},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  written to {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
