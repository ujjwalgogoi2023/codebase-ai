# CodebaseAI — RAG-Powered Codebase Assistant

Ask natural-language questions about a codebase and get answers grounded in the
actual source, with file-and-line citations you can click through to the code.

```
"Where do we validate JWT tokens?"

  Token validation happens in the auth middleware [1], which extracts the
  bearer token before the request reaches any route handler [2].

  Sources
  [1] backend/middleware.py:18-39   (require_auth)
  [2] backend/routes.py:10-28       (register_routes)
```

Built to demonstrate a Retrieval-Augmented Generation pipeline end to end —
including the parts most RAG demos skip: knowing whether it actually works.

---

## Architecture

```
INGESTION                                    QUERY
─────────                                    ─────
GitHub / ZIP / local path                    Question
        │                                        │
        ▼                                        ▼
  Repository parser                        Query embedding
  (filter, detect language)                      │
        │                                        ▼
        ▼                                  Vector search  ──────┐
  Code-aware chunker                       (pgvector, HNSW,     │
  (AST / brace matching)                    cosine distance)    │
        │                                        │              │
        ▼                                        ▼              │
    Embeddings                             Top-10 candidates    │
        │                                        │              │
        ▼                                        ▼              │
  PostgreSQL + pgvector  ◄───────────────►   Reranker           │
  repositories → files → chunks            (listwise, LLM)      │
                                                 │              │
                                                 ▼              │
                                           Top-3 chunks         │
                                                 │              │
                                                 ▼              │
                                        LLM (grounded prompt)   │
                                                 │              │
                                                 ▼              │
                                     Answer + validated [n]  ◄──┘
                                          citations
```

**Why retrieve 10 but send 3?** Vector similarity tells you which chunks are
semantically *similar*; it doesn't tell you which chunk *answers the question*.
So we cast a wide net cheaply, then spend a slower, more accurate model on the
shortlist.

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11, FastAPI, SQLAlchemy 2.0 |
| Vector store | PostgreSQL 17 + pgvector 0.8 (HNSW, cosine) |
| Embeddings | `qwen3-embedding:0.6b` via Ollama, or OpenAI `text-embedding-3-small` |
| Generation | `qwen2.5:7b` via Ollama, or OpenAI `gpt-4o-mini` |
| Frontend | React 19, Vite, Tailwind 4 |
| Infra | Docker Compose |
| Tests | pytest — **330 tests** |

Providers are an `.env` change: `ollama` (free, local), `openai`, or `fake`
(offline, for tests). Every number in this README was measured on the local
models, on a laptop with a 6 GB RTX 4050.

No LangChain or LlamaIndex. Every stage — chunking, embedding, retrieval,
reranking, prompt assembly, citation validation — is written directly, because
the point was to understand the pipeline rather than to configure one.

---

## How it works

### 1. Parsing
Walks the repository and decides what is worth indexing. Prunes ignored
directories **in place during the walk**, so vendored trees are never descended
into. On one test repository this meant walking 3 files instead of 919 — the
other 916 were `node_modules`.

Virtual environments are detected by their `pyvenv.cfg` marker rather than by
name, because `venv/` and `.venv/` are conventions, not rules (`myenv/` is not).

### 2. Code-aware chunking
A chunk is the atomic unit of retrieval — nothing smaller can ever be
retrieved, so the boundary decides which questions are answerable at all.
Naive `text[i:i+500]` cuts mid-function, leaving one chunk with no ending and
another with no name; neither embeds as "authenticating a user".

- **Python** — the stdlib AST. Decorators are part of the function; large
  classes split into methods; unparseable files fall back to windowing rather
  than being dropped.
- **JS / TS / Java** — brace matching over string- and comment-masked lines, so
  `const s = "}}}"` can't move a boundary.
- **Markdown** — heading sections.

All language strategies feed one shared pipeline: *fill gaps → merge tiny →
split oversized*. Gap-filling guarantees the invariant that matters most:

> **No source line is ever lost.**

Verified on 1,287 real files (fastapi, sqlalchemy, openai, this backend):
14,032 chunks, zero lost lines. A chunker that silently drops lines produces a
system that says *"that code doesn't exist"* about code that does — the worst
RAG failure, because a wrong refusal is indistinguishable from a right one.

**Referenced definitions.** Cutting on syntax separates a function from the
constants it reads: `login()` is one chunk, `MAX_FAILED_ATTEMPTS = 5` is part
of another near the top of the file. Retrieval finds `login()`, and the model
sees `if failed >= MAX_FAILED_ATTEMPTS` with no number anywhere. So each Python
chunk also records the short module-level constants and helper functions
(up to 6 lines) it uses from the same file. They are shown to the model beside
the chunk, but are **not** embedded and **not** part of the chunk's line range,
so retrieval and citations are unchanged. A repository indexed before this
existed picks it up on its next re-index.

### 3. Embeddings
Batched, with per-input token truncation and jittered exponential backoff.
Results are sorted by the API's `index` field rather than trusting arrival
order — a silent mis-zip would attach every vector to the wrong chunk.

A SQLite cache keyed by `(model, text)` makes re-indexing free during
development. The model is in the key, so switching models never serves stale
vectors.

### 4. Storage
One database, one transaction, one consistency story. A dedicated vector store
would mean "delete a repository and everything under it" is a distributed
delete across two systems with no shared transaction — and any failure between
them leaves orphaned vectors that retrieval returns forever.

`repository_id`, `language` and `file_path` are denormalised onto `chunks` so a
filtered search needs no join.

**What the query plan actually shows.** The original reasoning was that the
column keeps per-repository search on the HNSW index. `EXPLAIN ANALYZE` says
otherwise at this scale: for "search inside repository N", Postgres (pgvector
0.8.6) never touches the HNSW index. It fetches that repository's rows through
the B-tree on `repository_id` — 1,888 for the largest indexed app — and sorts
them by exact distance. So per-repository search is currently **exact, with
perfect recall**, and still fast. The denormalised column pays off by making
that scan cheap and join-free.

HNSW only takes over when one repository is far larger. Then the filter is
applied *after* the graph walk, which examines about `ef_search` candidates
from every repository, so it can return fewer than `top_k` results or miss
closer ones. pgvector 0.8's `hnsw.iterative_scan` setting, or partitioning
`chunks` by repository, is the fix if that day comes.

### 5. Retrieval
Cosine distance via pgvector's `<=>`, with metadata filters (language, kind,
path prefix, minimum score) applied as ordinary SQL predicates in the same
query as the vector search.

Refuses to search if the repository was indexed with a **different embedding
model**. Vectors of equal dimension from different models still "work" — they
return noise with confident citations attached, with no error anywhere.

### 6. Generation
Three defences against invented code, in order of reliability:

1. **Empty retrieval never reaches the model.** Guaranteed correct refusal,
   zero cost, cannot fail.
2. The system prompt forbids outside knowledge and requires an explicit "not in
   this codebase".
3. Returned citations are validated (below).

Prompting alone is the *weakest* of the three, which is exactly why the other
two exist.

Conversation history is replayed so pronouns resolve, but **every question runs
its own search**. History is not a substitute for retrieval.

**Follow-ups are rewritten before anything else happens.** "Which constant
controls that?" cannot be searched, and replaying history alone was not enough:
after a turn about late fees, the right code was retrieved and the model still
answered about an unrelated constant in its third source. So a follow-up is
first rewritten by the model into one standalone question ("Which constant
controls the late fee per day?"), and that question is what gets searched,
reranked and answered. It costs one extra call per follow-up and fails soft: an
error, or a "rewrite" that is really an answer, falls back to prepending the
previous question. `QUERY_REWRITER=concat` keeps the old behaviour as a
control.

### 7. Citations
The model is shown numbered sources and cites `[1]`, `[2]` — it picks an index
rather than writing a path, so every marker maps back to an exact file and line
range. Markers pointing at sources that don't exist are removed from the
displayed answer and recorded in `invalid_markers`, with the model's raw output
preserved for auditing.

The interesting problem: answers about code contain code, and code contains
brackets. `items[0]` must not parse as citation 0 and get stripped. Fenced
blocks and inline spans are masked, and a marker preceded by an identifier or
`)` is treated as an index expression. Telling adjacent citations `[2][3]` from
a chained index `grid[1][2]` needs position, not a lookbehind.

### 8. Reranking
Listwise, not pointwise: one call ranking all candidates rather than one call
per candidate. Fails soft in every direction — malformed output, omitted
candidates, a hallucinated index or an outright exception all fall back to the
vector ordering rather than failing the question.

---

## Setup

```bash
git clone <your-repo-url> codebase-ai
cd codebase-ai
cp .env.example .env          # pick providers, see below
```

```bash
docker compose up -d          # PostgreSQL 17 + pgvector on port 5433
```

```bash
python -m venv .venv && .venv/Scripts/pip install -r backend/requirements.txt
```

```bash
.venv/Scripts/python -m uvicorn app.main:app --app-dir backend --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** (the UI) or **http://localhost:8000/docs** (the
generated API reference).

### Running locally for free (Ollama)

Install [Ollama](https://ollama.com), then pull the two models and build the
8k-context variant of the LLM:

```bash
ollama pull qwen3-embedding:0.6b
```

```bash
ollama pull qwen2.5:7b
```

```bash
ollama create qwen2.5-8k -f ollama/qwen2.5-8k.Modelfile
```

and set in `.env`:

```bash
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_DIM=1024
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5-8k
```

Changing `EMBEDDING_DIM` changes the width of the vector column, so recreate the
schema afterwards with `python scripts/init_db.py --reset`.

**One untested gain.** `qwen3-embedding` was trained with a task instruction in
front of each *query* (`{Instruction} {Query}`; documents are embedded as-is),
per its technical report ([arXiv:2506.05176](https://arxiv.org/abs/2506.05176)).
CodebaseAI currently sends bare questions. Retrieval works without the
instruction — every number in this README was measured that way — but adding
one, such as "Given a question about a codebase, retrieve the code that answers
it", is likely a small free improvement. It has not been measured yet.

**Why the 8k variant.** Ollama defaults to a 4096-token context and silently
drops the *start* of a longer prompt — which is where the grounding instructions
live. `MAX_CONTEXT_TOKENS` allows up to 8000 tokens of retrieved code, so the
default window can quietly turn a grounded prompt into an ungrounded one.

On the 6 GB GPU above, indexing this backend (586 chunks) takes about 30 s and
a full question — search, rerank, answer — about 15–18 s.

### Running fully offline (fakes)

Everything also runs with deterministic fake providers and no model at all:

```bash
EMBEDDING_PROVIDER=fake
LLM_PROVIDER=fake
```

The full pipeline, API and UI work; only the *semantics* are absent, since fake
embeddings are random vectors. This is what keeps the test suite free and
network-independent; the tests pin these providers themselves, whatever `.env`
says.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/repositories` | register from a local path or GitHub URL |
| `POST` | `/repositories/upload` | register from a ZIP upload |
| `GET` | `/repositories/{id}/status` | poll indexing progress |
| `POST` | `/repositories/{id}/index` | re-index |
| `GET` | `/repositories/{id}/files` | list indexed files |
| `GET` | `/repositories/{id}/search` | **retrieval only, no LLM** |
| `POST` | `/chat` | ask a question |
| `GET` | `/conversations/{id}` | conversation history |
| `GET` | `/health` | liveness + the providers actually in use |

**Long-running indexing.** `POST` returns an id and `index_status: "pending"`
immediately; work runs in a background task; the client polls `/status`. Status
lives in the database, not process memory, so it survives a reconnect and a
crash leaves a row saying `failed` with the reason rather than one stuck on
`indexing`. At higher concurrency this would move to a real queue (Celery/RQ)
for retries and durability — a deliberate deferral, not an oversight.

**`/search` exists for debugging.** When an answer is wrong there are only two
candidates: retrieval fetched the wrong code, or the LLM misread the right
code. They need completely different fixes, and this endpoint separates them.

---

## Evaluation

> *"How did you know your RAG system was working?"*

Two labelled datasets. Questions are phrased the way a developer would ask
them rather than by quoting identifiers, so a hit requires semantic retrieval
rather than keyword overlap.

- [`evaluation/dataset.json`](evaluation/dataset.json) — 55 questions about this
  repository's own backend.
- [`evaluation/dataset_flask.json`](evaluation/dataset_flask.json) — 45
  questions about [Flask](https://github.com/pallets/flask) 3.1.3, written
  **before** any configuration was run against it, so nothing was tuned to it.
  It is the held-out check on conclusions drawn from the first set.

```bash
python evaluation/evaluate_retrieval.py
```

```bash
python evaluation/run_experiments.py
```

```bash
git clone --depth 1 --branch 3.1.3 https://github.com/pallets/flask data/eval_repos/flask
```

```bash
python evaluation/run_experiments.py --dataset evaluation/dataset_flask.json
```

**Metrics.** Top-1/3/5 accuracy (did an expected file appear in the top K?) and
MRR. Top-3 matters most, because `RERANK_TOP_K=3` — Top-3 accuracy *is* the
ceiling on how often the LLM can possibly answer correctly. MRR is reported
alongside because it's sensitive to *where* the hit landed: it can distinguish a
change that moved the answer from rank 4 to rank 2 from one that did nothing.

Retrieval is measured rather than answer quality for three reasons: retrieval is
the ceiling (no model can answer from a chunk that never reached the prompt), it
has cheap exact ground truth, and it's deterministic, so a difference between
runs is a real difference in the system rather than sampling noise.

### Results

Local models: `qwen3-embedding:0.6b` for retrieval, `qwen2.5:7b` for
reranking. Full tables, including the top-K and chunk-size sweeps, are in
[`evaluation/results/`](evaluation/results/).

**This backend** (55 questions)

| Configuration | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|
| naive chunking | 23.6% | 72.7% | 85.5% | 0.475 |
| code-aware chunking | 23.6% | 54.5% | 72.7% | 0.435 |
| code-aware + reranking | **38.2%** | 70.9% | 78.2% | **0.552** |

**Flask 3.1.3, held out** (45 questions)

| Configuration | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|
| naive chunking | 53.3% | 68.9% | 84.4% | 0.650 |
| code-aware chunking | 60.0% | 84.4% | 91.1% | 0.729 |
| code-aware + reranking | **77.8%** | **91.1%** | **93.3%** | **0.845** |

What the numbers say:

- **Reranking is the one consistent win.** Top-1 rises by 15–18 points on both
  repositories. Vector similarity finds the right neighbourhood; the reranker
  picks the right house.
- **Code-aware chunking won on Flask and lost on this backend.** The difference
  is chunk size. Flask's functions carry their own docstrings, so a function is
  a self-describing unit (median chunk 14 lines). This backend explains its
  design in long module docstrings, which become chunks separate from the code
  they describe, and half its code-aware chunks are 8 lines or fewer — too
  little text to say what the file is for. The small-chunk merge only fires
  when *both* neighbours are under 5 lines, so short functions stay isolated.
- **Packing small chunks together was tried and not adopted.** Greedily merging
  neighbouring functions up to 60 lines lifted this backend to 40.0% Top-1 —
  but 60 was chosen by looking at those results, and on held-out Flask the
  effect was mixed (Top-1 up 6.7 points, Top-3 and Top-5 down 2.2). One
  question is worth about 2 points here, so that is not evidence of a real
  improvement.
- **Test files are the biggest single factor.** They take about half of all
  top-5 slots. Excluding `tests/` from the index raised code-aware Top-1 from
  23.6% to 61.8% on this backend and from 60.0% to 82.2% on Flask. Part of that
  is labelling — only source files count as correct, so a test exercising the
  right behaviour scores as a miss — but an assistant that answers "how does X
  work?" from a test is still giving a worse answer.

The harness keeps its own results honest: `EvalReport.is_meaningful` is
`False` for any run involving a fake provider, and both the terminal report and
the generated Markdown carry a disclaimer automatically, so a meaningless run
cannot be mistaken for a result.

### Answer quality

Retrieval accuracy is the ceiling, not the product. The answers themselves were
tested end to end through `/chat` on a private 247-file Python and TypeScript
application: 14 questions whose answers were checked by hand against its
source, plus 5 about features it does not have (Stripe, GraphQL, Google OAuth,
Elasticsearch, and a class-timetable feature that a migration had removed).

| | Before the two fixes | After |
|---|---:|---:|
| Right file among the 3 chunks sent | 14/14 | 14/14 |
| Right file cited | 13/14 | 13/14 |
| **Key facts in the answer** | **8/14** | **11/14** |
| Said "not shown" when it was there | 3/14 | 1/14 |
| Correctly refused a missing feature | 5/5 | 5/5 |
| Fabricated citations | 0 | 0 |

- **Missing values were reported, not guessed.** Where the number was not in
  the sources, the answer named the constant it could see
  (`MAX_FAILED_ATTEMPTS`, `settings.otp_ttl_minutes`) and said its value was
  not shown.
- **Referenced definitions fixed the most common gap**: retrieval found the
  right function, but the value the question asked for lived in a constant in
  another chunk. Login lockout, code generation and the release cooldown are
  now answered with their actual values.
- **What is still missed**: a value defined in a *different* file (token
  lifetimes live in `config.py`), and supporting code that does not make the
  3-source cut. Sending 5 sources instead of 3 recovered none of them; the
  extra slots filled with tests and docs.
- **Follow-up rewriting** fixed the failing case above. On six realistic
  follow-ups the two strategies tied 5/6, because a follow-up that names its
  subject ("...if Redis is unavailable?") already searches well. Rewriting
  matters for the ones that do not.

#### Run these tests yourself

Three scripts in `evaluation/`, all against real models (start the backend
first for the first two):

```bash
python evaluation/system_test.py
```

```bash
python evaluation/answer_quality.py evaluation/answers_codebaseai.json
```

```bash
python evaluation/followup_ab.py evaluation/answers_codebaseai.json
```

- **`system_test.py`** — 33 black-box checks of the running API: indexing,
  search, grounded answers, follow-ups, refusals, input validation, ZIP upload
  including a Zip Slip archive, re-index and delete. It builds a throwaway
  repository full of facts no model could know (a 37-rupee late fee), so a
  correct answer proves it came from retrieval. Exits non-zero on any failure.
- **`answer_quality.py`** — scores `/chat` answers against a question set: was
  the right file sent and cited, does the answer contain the key facts, and
  does it refuse features that do not exist. Results are written next to the
  question set, so a set kept in `data/` (gitignored) stays private.
- **`followup_ab.py`** — the same follow-up questions under
  `QUERY_REWRITER=concat` and `llm`, from identical history.

`evaluation/answers_codebaseai.json` asks about this repository's own backend
(index it as `codebaseai-backend` first). On it the right file reached the
model for 6 of 10 questions and the key fact appeared in 5; all 5 missing
features were refused, and the follow-up strategies tied 4/4. The misses are
the test-file problem again: `test_zip_bomb_is_rejected_before_extraction`
and its neighbours took the slots meant for the code that defines the values,
and 64% of this backend's chunks are tests.

### How much retrieval depends on names

What happens on a poorly written codebase — meaningless names, no docstrings,
no comments? To measure it rather than guess, Flask's 24 source files were
obfuscated with Python's AST: all 701 names Flask defines (classes, functions,
methods, parameters, variables, attributes) renamed consistently across files,
and all docstrings, comments and type hints removed. Library calls (werkzeug,
jinja2, click, itsdangerous) and string literals were kept, as they would be in
real badly written code. The same 45 held-out Flask questions were then run on
source files only, so test files could not crowd the results.

```python
# before
def should_set_cookie(self, app: Flask, session: SessionMixin) -> bool:
    """Used by session backends to determine if a ``Set-Cookie`` header ..."""
    return session.modified or (
        session.permanent and app.config["SESSION_REFRESH_EACH_REQUEST"]
    )

# after
def f203(self, f45, f196):
    return f196.v261 or (f196.f167 and f45.v126['SESSION_REFRESH_EACH_REQUEST'])
```

| Condition | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|
| **A** · clean | 82.2% | 95.6% | 97.8% | 0.889 |
| A · clean, reranked | **93.3%** | **100%** | **100%** | **0.963** |
| **B** · names, docstrings, comments removed | 75.6% | 86.7% | 91.1% | 0.828 |
| B · reranked | 73.3% | 88.9% | 91.1% | 0.818 |
| **C** · B, plus files renamed `m01.py`… | 60.0% | 82.2% | 91.1% | 0.727 |
| C · reranked | 60.0% | 84.4% | 91.1% | 0.734 |
| **D** · C, plus an LLM summary embedded with each chunk | 66.7% | 80.0% | 88.9% | 0.757 |
| D · reranked | 66.7% | 86.7% | 91.1% | 0.778 |

One question is about 2 points, so differences of a few points are one or two
questions.

- **It degrades, it does not collapse.** With reranking, Top-1 fell from 93%
  to 60% and Top-3 from 100% to 84%, but the right file still reached the top
  5 for nine questions in ten. Library calls and string literals carry
  meaning that renaming cannot remove.
- **File paths are a large signal on their own.** Renaming the files cost 13–16
  points of Top-1 beyond renaming everything inside them — `sessions.py` says a
  lot. It is also the cheapest thing to get right in a real codebase.
- **The reranker needs meaning to read.** It added 11 points of Top-1 on clean
  code and nothing once names were gone: it ranks from code previews, and
  `f196.v261` gives it nothing to judge.
- **LLM chunk summaries recovered little.** A few points of Top-1 and MRR,
  within the noise, with Top-3 slightly down before reranking. A summary of
  meaningless code is generic; it cannot restore intent that is no longer in
  the text. The test was tilted *in the summaries' favour* — the summarising
  model recognised the obfuscated code as Flask — so on genuinely unfamiliar
  code expect even less. They are also slow locally: 251 chunks took 32
  minutes.

So the levers that matter for hard-to-read code are mostly in the code itself
— meaningful file names and docstrings — plus a keyword index for the string
literals that survive obfuscation, rather than summarising chunks after the
fact. This was a one-off experiment: its obfuscator and runner live outside
this repository and are not part of the evaluation suite.

---

## Design decisions

**PostgreSQL + pgvector over a dedicated vector database.** The system needs
vectors *and* relational data — repositories own files, files own chunks,
conversations own messages, all with foreign keys and cascade deletes. One
transaction beats a distributed delete with no shared transaction.

**HNSW over IVFFlat.** No training pass, works on an empty table, so indexing
order is irrelevant. `vector_cosine_ops` must match the `<=>` operator or
Postgres silently ignores the index.

**Code-aware chunking over fixed windows — measured, not assumed.** Syntactic
boundaries produce chunks whose embeddings mean what the code means, and on
held-out Flask that won by 15 points of Top-3. On this repository it lost,
because its chunks came out too small (see [Results](#results)). Both
strategies stay in the codebase so the comparison can be re-run on any corpus.

**One repository row per indexing run**, keyed by `(name, model, strategy)`, so
naive and code-aware indexes coexist and the experiment doesn't require
re-indexing between every question.

**Temperature 0.** A grounded answer should be reproducible, or the evaluation
measures sampling noise as much as retrieval quality.

**Provider abstractions with offline fakes.** Tests never spend money or need
the network, and switching providers is an `.env` change rather than a refactor.

---

## Security

Source acquisition is the only place untrusted input enters, and it's guarded:

- **Zip Slip** — every archive member is validated against the *resolved*
  destination. `ZipFile.extractall()` writes wherever the archive says.
- **Zip bombs** — uncompressed size and member count are capped from the
  headers, before extraction.
- Absolute paths, drive letters and symlink entries are rejected or skipped.
- **SSRF** — clone URLs must be `https` on an allowlisted host, blocking
  `file://`, `ssh://` and requests to internal or cloud-metadata addresses. Git
  runs as an argument list (never a shell string) with `GIT_TERMINAL_PROMPT=0`
  and a timeout, in a minimal environment: an allowlist of `PATH`,
  `SYSTEMROOT` (without which Windows git cannot resolve any host) and the
  proxy variables, so stray `GIT_*` settings cannot change what a clone does.

Secrets live only in `.env`, which is gitignored and never committed.

---

## Project layout

```
backend/
  app/
    api/          repositories.py, chat.py, deps.py
    db/           engine, session, schema creation
    models/       SQLAlchemy schema (incl. the pgvector column)
    schemas/      Pydantic request/response models
    services/     repository_parser, chunker, embeddings, indexer,
                  retriever, reranker, generator, citations, llm,
                  sources, evaluation
    config.py     typed settings from .env
    main.py       FastAPI app
  tests/          330 tests
frontend/         React + Vite + Tailwind
evaluation/       datasets (own backend, Flask), harness, experiment runner,
                  answer-quality and API system tests, results/
ollama/           Modelfile for the 8k-context local LLM
scripts/          per-stage CLIs (parse, chunk, index, search, ask)
docker-compose.yml
```

Every pipeline stage has a CLI so it can be inspected on its own:

```bash
python scripts/parse_repo.py <path>
python scripts/chunk_repo.py <path> --compare
python scripts/index_repo.py <path> --name my-app
python scripts/search.py 1 "where is auth?" --show-content
python scripts/ask.py 1 "how does indexing work?" --show-prompt
python scripts/audit_chunk_coverage.py <path>
```

`ask.py --show-prompt` prints the exact text the model receives. Worth looking
at once: RAG is often described as though something clever happens at
generation time, and the prompt makes clear that the cleverness all happened
earlier, in retrieval.

---

## Tests

```bash
cd backend && ../.venv/Scripts/python -m pytest
```

330 tests. Database-backed tests use a separate `_test` database, created on
demand, and the suite **refuses to run** against any database whose name doesn't
end in `_test` — an earlier version truncated the development database on every
run.
