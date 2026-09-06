"""Dev CLI for pipeline stage 3 — prove embeddings capture meaning.

Usage:
    python scripts/embed_demo.py                 # semantic similarity demo
    python scripts/embed_demo.py --repo <path>   # cost estimate for a real repo

The demo asks a question in plain English and ranks code snippets by cosine
similarity. The point to watch: the best match shares NO keywords with the
question. That is the whole reason RAG beats grep, and it is worth seeing with
your own key rather than taking on faith.

Costs a fraction of a cent (a handful of short inputs).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.services.chunker import chunk_files  # noqa: E402
from app.services.embeddings import (  # noqa: E402
    cosine_similarity,
    estimate_cost,
    get_embedding_provider,
)
from app.services.repository_parser import parse_repository  # noqa: E402

SNIPPETS = {
    "verify_credentials": (
        "def verify_credentials(email, secret):\n"
        "    account = Account.query.filter_by(email=email).first()\n"
        "    if account is None:\n"
        "        return None\n"
        "    return account if bcrypt.checkpw(secret, account.digest) else None"
    ),
    "resize_thumbnail": (
        "def resize_thumbnail(image, width, height):\n"
        "    img = Image.open(image)\n"
        "    img.thumbnail((width, height))\n"
        "    return img"
    ),
    "issue_token": (
        "def issue_token(user_id):\n"
        "    payload = {'sub': user_id, 'exp': time.time() + 3600}\n"
        "    return jwt.encode(payload, SECRET, algorithm='HS256')"
    ),
    "connect_database": (
        "engine = create_engine(DATABASE_URL, pool_size=10)\n"
        "SessionLocal = sessionmaker(bind=engine)"
    ),
    "send_invoice_email": (
        "def send_invoice_email(order):\n"
        "    body = render_template('invoice.html', order=order)\n"
        "    smtp.send(order.customer.email, 'Your invoice', body)"
    ),
}

QUERIES = [
    "where do we check a user's password when they log in?",
    "how are JSON web tokens created?",
    "where is the database connection set up?",
]


def run_demo() -> int:
    settings = get_settings()
    provider = get_embedding_provider(settings)

    print(f"\nProvider: {settings.embedding_provider}  Model: {provider.model}  "
          f"Dim: {provider.dimension}")

    names = list(SNIPPETS)
    texts = [SNIPPETS[n] for n in names]

    estimate = estimate_cost(texts + QUERIES, model=provider.model)
    print(f"Embedding {estimate['texts']} inputs, ~{estimate['tokens']} tokens, "
          f"~${estimate['usd']}\n")

    snippet_vectors = provider.embed(texts)
    query_vectors = provider.embed(QUERIES)

    for query, qv in zip(QUERIES, query_vectors):
        scored = sorted(
            ((cosine_similarity(qv, sv), name) for sv, name in zip(snippet_vectors, names)),
            reverse=True,
        )
        print(f'  Q: "{query}"')
        for rank, (score, name) in enumerate(scored, start=1):
            marker = "->" if rank == 1 else "  "
            bar = "#" * int(score * 40)
            print(f"   {marker} {score:.3f} {name:<20} {bar}")
        print()

    hits = getattr(provider, "hits", None)
    if hits is not None:
        print(f"  cache: {provider.hits} hits, {provider.misses} misses")

    print(
        "  Note how the top match for the password question is"
        "\n  `verify_credentials`, which contains neither the word 'password'"
        "\n  nor 'log in'. Keyword search would have missed it entirely."
    )
    return 0


def run_estimate(repo: str) -> int:
    settings = get_settings()
    files, stats = parse_repository(repo, max_file_bytes=settings.max_file_bytes)
    chunks = chunk_files(files, max_chunk_lines=settings.max_chunk_lines)
    texts = [c.embedding_text for c in chunks]

    estimate = estimate_cost(texts, model=settings.embedding_model)
    print(f"\nRepository: {Path(repo).resolve()}")
    print(f"  files      {stats.kept}")
    print(f"  chunks     {estimate['texts']}")
    print(f"  tokens     {estimate['tokens']:,}")
    print(f"  model      {estimate['model']}")
    print(f"  est. cost  ${estimate['usd']}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Embedding demo / cost estimator.")
    ap.add_argument("--repo", help="estimate indexing cost for a repository")
    args = ap.parse_args()

    return run_estimate(args.repo) if args.repo else run_demo()


if __name__ == "__main__":
    raise SystemExit(main())
