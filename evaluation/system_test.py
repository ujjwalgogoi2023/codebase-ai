"""Black-box system test of the running API, on real models.

Usage:
    python evaluation/system_test.py
    python evaluation/system_test.py --api http://127.0.0.1:8000

Start the backend first, with real providers (ollama or openai). The unit
suite runs on fakes and mocks; this is the check that the assembled system
works: indexing, search, grounded answers, follow-ups, refusals, input
validation, ZIP upload including a hostile archive, re-index and delete.

It builds a tiny repository full of facts no model could know from training
(a 37-rupee late fee, a 7-attempt lockout), so a correct answer proves the
answer came from retrieval rather than from the model's memory. Everything it
creates is deleted at the end. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import httpx

FILES = {
    "billing/fees.py": '''"""Library billing rules."""

LATE_FEE_PER_DAY = 37  # rupees
GRACE_DAYS = 3


def compute_late_fee(days_overdue: int) -> int:
    """Fee owed for a book returned late. The first GRACE_DAYS are free."""
    billable = max(0, days_overdue - GRACE_DAYS)
    return billable * LATE_FEE_PER_DAY
''',
    "auth/lockout.py": '''"""Account lockout after repeated failed logins."""

MAX_FAILED_ATTEMPTS = 7
LOCKOUT_MINUTES = 45


def should_lock(failed_attempts: int) -> bool:
    return failed_attempts >= MAX_FAILED_ATTEMPTS
''',
    "web/src/maskEmail.ts": '''// Hide most of an email address before showing it in the admin panel.
export function maskEmail(email: string): string {
  const [user, domain] = email.split("@");
  if (!domain) return "***";
  return user.slice(0, 2) + "*****@" + domain;
}
''',
    "README.md": "# Planted test repo\n\nA tiny library system used to test grounding.\n",
}

REFUSAL_MARKERS = ("not", "no ", "doesn't", "isn't")

results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def wait_ready(client: httpx.Client, repository_id: int, timeout: float = 300) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/repositories/{repository_id}/status").json()
        if status["index_status"] in ("ready", "failed"):
            return status
        time.sleep(1)
    raise TimeoutError(f"repository {repository_id} never finished indexing")


def chat(client: httpx.Client, repository_id: int, question: str, conversation_id=None) -> dict:
    body = {"repository_id": repository_id, "question": question}
    if conversation_id:
        body["conversation_id"] = conversation_id
    response = client.post("/chat", json=body)
    response.raise_for_status()
    return response.json()


def short(text: str) -> str:
    return " ".join(text.split())[:150]


def run(client: httpx.Client, workdir: Path) -> None:
    repo_dir = workdir / "planted_repo"
    for rel, text in FILES.items():
        path = repo_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    created: list[int] = []
    try:
        print("\n[1] health")
        health = client.get("/health").json()
        check("database reachable", health["database"] == "ok")
        check("real providers in use",
              "fake" not in (health["embedding_provider"], health["llm_provider"]),
              f"{health['embedding_provider']} / {health['llm_provider']}")

        print("\n[2] register a local path and index it")
        r = client.post("/repositories", json={"path": str(repo_dir), "name": "system-test"})
        check("register returns 201", r.status_code == 201, str(r.status_code))
        repo_id = r.json()["id"]
        created.append(repo_id)
        status = wait_ready(client, repo_id)
        check("indexing reaches ready", status["index_status"] == "ready", json.dumps(status))
        repo = client.get(f"/repositories/{repo_id}").json()
        check("all 4 files indexed (py, ts, md)", repo["file_count"] == 4,
              f"files={repo['file_count']} chunks={repo['chunk_count']}")
        languages = {f["language"] for f in client.get(f"/repositories/{repo_id}/files").json()}
        check("languages detected", languages == {"python", "typescript", "markdown"}, str(languages))

        print("\n[3] retrieval only (/search)")
        s = client.get(f"/repositories/{repo_id}/search",
                       params={"q": "how much do we charge for overdue books"}).json()
        top = s["results"][0]
        check("late-fee code ranks first", top["file_path"] == "billing/fees.py",
              f"{top['file_path']} {top['score']:.3f}")
        s = client.get(f"/repositories/{repo_id}/search",
                       params={"q": "hide part of an email address", "language": "typescript"}).json()
        check("language filter returns only TypeScript",
              s["count"] > 0 and all(x["language"] == "typescript" for x in s["results"]))

        print("\n[4] grounded answers about planted facts")
        a = chat(client, repo_id, "How much is the late fee per day, and are any days free?")
        check("states the planted fee (37)", "37" in a["answer"], short(a["answer"]))
        check("mentions the 3 grace days", "3" in a["answer"] or "three" in a["answer"].lower())
        check("cites billing/fees.py",
              any(c["file_path"] == "billing/fees.py" for c in a["citations"]),
              str([c["citation"] for c in a["citations"]]))
        check("marked grounded, no invalid markers", a["is_grounded"] and not a["invalid_markers"])
        b = chat(client, repo_id, "After how many failed logins is an account locked, and for how long?")
        check("states 7 attempts and 45 minutes", "7" in b["answer"] and "45" in b["answer"],
              short(b["answer"]))
        c = chat(client, repo_id, "How is an email address masked before it is shown to admins?")
        check("TypeScript function answered and cited",
              any(x["file_path"] == "web/src/maskEmail.ts" for x in c["citations"]), short(c["answer"]))

        print("\n[5] conversation history")
        f = chat(client, repo_id, "Which constant controls that?", conversation_id=a["conversation_id"])
        check("follow-up resolves 'that' to the late fee",
              "LATE_FEE_PER_DAY" in f["answer"] or "GRACE_DAYS" in f["answer"], short(f["answer"]))
        conversation = client.get(f"/conversations/{a['conversation_id']}").json()
        check("conversation stores 4 messages", len(conversation["messages"]) == 4,
              str(len(conversation["messages"])))

        print("\n[6] refusing what is not there")
        n = chat(client, repo_id, "How does the app send password reset emails through SendGrid?")
        lowered = n["answer"].lower()
        check("does not invent SendGrid code",
              any(m in lowered for m in REFUSAL_MARKERS) and "sendgrid.send" not in lowered,
              short(n["answer"]))
        check("no fabricated citations", not n["invalid_markers"])

        print("\n[7] input validation and security")
        for label, body in [
            ("file:// clone URL rejected", {"url": "file:///C:/Windows/win.ini"}),
            ("cloud metadata address rejected", {"url": "http://169.254.169.254/latest/meta-data"}),
            ("non-allowlisted host rejected", {"url": "https://evil.example.com/owner/repo"}),
            ("both url and path rejected", {"url": "https://github.com/pallets/flask", "path": str(repo_dir)}),
            ("missing directory rejected", {"path": str(workdir / "does-not-exist")}),
        ]:
            r = client.post("/repositories", json=body)
            check(label, r.status_code == 422, str(r.status_code))
        r = client.post("/chat", json={"repository_id": repo_id, "question": ""})
        check("empty question rejected", r.status_code == 422, str(r.status_code))
        r = client.post("/chat", json={"repository_id": 999_999_999, "question": "hi"})
        check("unknown repository is 404", r.status_code == 404, str(r.status_code))

        print("\n[8] ZIP upload, including a hostile archive")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as z:
            for rel, text in FILES.items():
                z.writestr(f"planted/{rel}", text)
        r = client.post("/repositories/upload",
                        files={"archive": ("planted.zip", archive.getvalue(), "application/zip")},
                        data={"name": "system-test-zip"})
        check("ZIP upload accepted", r.status_code == 201, str(r.status_code))
        zip_id = r.json()["id"]
        created.append(zip_id)
        zip_status = wait_ready(client, zip_id)
        zip_repo = client.get(f"/repositories/{zip_id}").json()
        check("ZIP indexes to the same chunks as the folder",
              zip_status["index_status"] == "ready" and zip_repo["chunk_count"] == repo["chunk_count"],
              f"zip={zip_repo['chunk_count']} folder={repo['chunk_count']}")

        evil = io.BytesIO()
        with zipfile.ZipFile(evil, "w") as z:
            z.writestr("ok.py", "x = 1\n")
            z.writestr("../../../../zipslip_canary.txt", "pwned")
        r = client.post("/repositories/upload",
                        files={"archive": ("evil.zip", evil.getvalue(), "application/zip")})
        check("Zip Slip archive rejected", r.status_code == 422, str(r.status_code))
        if r.status_code == 201:
            created.append(r.json()["id"])
        r = client.post("/repositories/upload",
                        files={"archive": ("notzip.zip", b"definitely not a zip", "application/zip")})
        check("non-ZIP upload rejected", r.status_code == 422, str(r.status_code))

        print("\n[9] re-index and delete")
        r = client.post(f"/repositories/{repo_id}/index")
        check("re-index accepted", r.status_code == 202, str(r.status_code))
        started = time.time()
        check("re-index completes", wait_ready(client, repo_id)["index_status"] == "ready",
              f"{time.time() - started:.1f}s")
        client.delete(f"/repositories/{repo_id}")
        created.remove(repo_id)
        check("deleted repository is gone", client.get(f"/repositories/{repo_id}").status_code == 404)
        check("its conversations were deleted too",
              client.get(f"/conversations/{a['conversation_id']}").status_code == 404)
    finally:
        for repository_id in created:
            client.delete(f"/repositories/{repository_id}")


def main() -> int:
    ap = argparse.ArgumentParser(description="System test against the running API.")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    client = httpx.Client(base_url=args.api, timeout=300)
    try:
        client.get("/health")
    except httpx.ConnectError:
        print(f"Cannot reach {args.api}. Start the backend first.", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as workdir:
        run(client, Path(workdir))

    passed = sum(ok for _, ok in results)
    print(f"\n  {passed}/{len(results)} checks passed\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
