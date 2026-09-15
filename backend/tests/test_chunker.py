"""Tests for code-aware chunking (RAG pipeline stage 2).

The most important test here is `test_no_source_line_is_ever_lost`. A chunker
that silently drops lines produces a system that confidently answers "that code
does not exist" about code that does exist — the worst possible RAG failure,
because it looks like a correct grounded refusal.
"""

import textwrap

import pytest

from app.services.chunker import (
    CodeChunk,
    chunk_file,
    chunk_files,
)
from app.services.repository_parser import SourceFile


def make_file(content: str, path: str = "a.py", language: str = "python") -> SourceFile:
    content = textwrap.dedent(content).lstrip("\n")
    return SourceFile(
        path=path,
        language=language,
        content=content,
        num_lines=len(content.splitlines()),
        size_bytes=len(content.encode()),
    )


def by_symbol(chunks: list[CodeChunk]) -> dict[str, CodeChunk]:
    return {c.symbol: c for c in chunks if c.symbol}


# ---------------------------------------------------------------------------
# The core invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "language,path,source",
    [
        (
            "python",
            "auth.py",
            """
            import os
            from db import session

            SECRET = os.environ["SECRET"]


            def login(user):
                return True


            # a stray comment between definitions
            class Service:
                def run(self):
                    return 1
            """,
        ),
        (
            "javascript",
            "app.js",
            """
            import React from "react";

            const API = "/api";

            export function App() {
              return null;
            }

            class Store {
              save(x) {
                return x;
              }
            }
            """,
        ),
        (
            "java",
            "Service.java",
            """
            package com.example;

            import java.util.List;

            public class Service {
                private final Repo repo;

                public void run() {
                    repo.save();
                }
            }
            """,
        ),
        (
            "markdown",
            "README.md",
            """
            Intro paragraph before any heading.

            # Title

            Body text.

            ## Setup

            More body.
            """,
        ),
    ],
)
def test_no_source_line_is_ever_lost(language, path, source):
    """Every non-blank line must appear in at least one chunk."""
    sf = make_file(source, path=path, language=language)
    chunks = chunk_file(sf, min_chunk_lines=1)

    covered: set[int] = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line + 1))

    lines = sf.content.splitlines()
    missing = [
        i + 1
        for i, line in enumerate(lines)
        if line.strip() and (i + 1) not in covered
    ]
    assert not missing, f"{language}: lines dropped from the index: {missing}"


def test_chunks_do_not_overlap_when_nothing_is_oversized():
    sf = make_file(
        """
        def a():
            return 1


        def b():
            return 2
        """
    )
    chunks = chunk_file(sf, min_chunk_lines=1)
    ordered = sorted(chunks, key=lambda c: c.start_line)
    for prev, nxt in zip(ordered, ordered[1:]):
        assert prev.end_line < nxt.start_line, "chunks overlap unexpectedly"


# ---------------------------------------------------------------------------
# Python (AST)
# ---------------------------------------------------------------------------

def test_function_stays_whole():
    sf = make_file(
        """
        import bcrypt


        def authenticate_user(username, password):
            user = db.get_user(username)
            if not user:
                return None
            if not bcrypt.checkpw(password, user.hash):
                return None
            return user
        """
    )
    chunks = chunk_file(sf, min_chunk_lines=1)
    fn = by_symbol(chunks)["authenticate_user"]

    assert fn.kind == "function"
    assert fn.content.startswith("def authenticate_user")
    assert "return user" in fn.content
    # The whole body, not a fragment.
    assert fn.content.count("return") == 3


def test_decorators_are_part_of_the_function():
    sf = make_file(
        """
        @app.post("/login")
        @requires_auth
        def login():
            return "ok"
        """
    )
    fn = by_symbol(chunk_file(sf, min_chunk_lines=1))["login"]

    assert fn.start_line == 1, "decorator was cut off from its function"
    assert "@app.post" in fn.content
    assert "@requires_auth" in fn.content


def test_small_class_is_kept_whole():
    sf = make_file(
        """
        class User:
            def __init__(self, name):
                self.name = name

            def greet(self):
                return self.name
        """
    )
    symbols = by_symbol(chunk_file(sf, min_chunk_lines=1))

    assert "User" in symbols
    assert symbols["User"].kind == "class"
    assert "User.greet" not in symbols, "small class should not be split"


def test_large_class_is_split_into_methods():
    body = "\n".join(
        f"    def method_{i}(self):\n" + "\n".join(f"        x{j} = {j}" for j in range(8))
        for i in range(6)
    )
    sf = make_file("class Big:\n" + body)

    symbols = by_symbol(chunk_file(sf, max_chunk_lines=20, min_chunk_lines=1))

    assert "Big" not in symbols, "oversized class should not be one chunk"
    assert "Big.method_0" in symbols
    assert "Big.method_5" in symbols
    assert symbols["Big.method_0"].kind == "method"


def test_unparseable_python_falls_back_to_windows():
    sf = make_file(
        """
        def broken(:
            this is not python at all ]]]
        """
    )
    chunks = chunk_file(sf, min_chunk_lines=1)

    assert chunks, "a syntactically broken file must still be indexed"
    assert all(c.kind == "window" for c in chunks)


# ---------------------------------------------------------------------------
# JavaScript / TypeScript (brace matching)
# ---------------------------------------------------------------------------

def test_javascript_declaration_forms():
    sf = make_file(
        """
        export function handler(req, res) {
          res.send("ok");
        }

        const validateToken = (token) => {
          return jwt.verify(token);
        };

        class Store {
          save(item) {
            return item;
          }
        }
        """,
        path="app.js",
        language="javascript",
    )
    symbols = by_symbol(chunk_file(sf, min_chunk_lines=1))

    assert "handler" in symbols
    assert "validateToken" in symbols
    assert "Store" in symbols
    assert "jwt.verify" in symbols["validateToken"].content


def test_braces_inside_strings_and_comments_do_not_break_boundaries():
    """The masking pass exists for exactly this."""
    sf = make_file(
        """
        const tpl = "function fake() { { {";
        const re = '}}}';
        // } } } a comment full of braces
        /* } } block comment } */
        function real(a) {
          const inner = "}";
          return a;
        }
        """,
        path="tricky.js",
        language="javascript",
    )
    fn = by_symbol(chunk_file(sf, min_chunk_lines=1))["real"]

    assert "return a;" in fn.content, "function body was truncated by a fake brace"
    assert fn.content.rstrip().endswith("}")


def test_typescript_annotations():
    sf = make_file(
        """
        export const getUser = async (id: string): Promise<User> => {
          return api.get(`/users/${id}`);
        };
        """,
        path="api.ts",
        language="typescript",
    )
    assert "getUser" in by_symbol(chunk_file(sf, min_chunk_lines=1))


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------

def test_java_class_and_methods():
    sf = make_file(
        """
        package com.example;

        public class OrderService {
            public Order createOrder(Cart cart) throws PaymentException {
                return repo.save(cart);
            }

            void cancel(long id) {
                repo.delete(id);
            }
        }
        """,
        path="OrderService.java",
        language="java",
    )
    symbols = by_symbol(chunk_file(sf, max_chunk_lines=8, min_chunk_lines=1))

    assert "OrderService.createOrder" in symbols
    assert "OrderService.cancel" in symbols, "package-private method was missed"


def test_java_control_flow_is_not_mistaken_for_a_method():
    sf = make_file(
        """
        public class Runner {
            public void go() {
                if (ready()) {
                    run();
                }
                Runnable r = new Runnable() {
                    public void run() {}
                };
            }
        }
        """,
        path="Runner.java",
        language="java",
    )
    symbols = by_symbol(chunk_file(sf, max_chunk_lines=6, min_chunk_lines=1))

    assert not any(s.endswith(".if") for s in symbols)
    assert not any(s.endswith(".Runnable") for s in symbols)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def test_markdown_splits_on_headings():
    sf = make_file(
        """
        # CodebaseAI

        Overview text.

        ## Setup

        Run docker compose up.

        ## API

        POST /chat
        """,
        path="README.md",
        language="markdown",
    )
    symbols = by_symbol(chunk_file(sf, min_chunk_lines=1))

    assert "CodebaseAI" in symbols
    assert "Setup" in symbols
    assert "docker compose up" in symbols["Setup"].content
    assert "POST /chat" in symbols["API"].content


# ---------------------------------------------------------------------------
# Size handling: the two ends of the tradeoff
# ---------------------------------------------------------------------------

def test_oversized_unit_is_windowed_with_overlap():
    sf = make_file("def huge():\n" + "\n".join(f"    x{i} = {i}" for i in range(100)))

    chunks = chunk_file(sf, max_chunk_lines=30, overlap_lines=5, min_chunk_lines=1)

    assert len(chunks) > 1, "oversized function should be split"
    assert all(c.num_lines <= 30 for c in chunks)
    ordered = sorted(chunks, key=lambda c: c.start_line)
    assert ordered[1].start_line <= ordered[0].end_line, "windows must overlap"


def test_tiny_adjacent_units_are_merged():
    sf = make_file(
        """
        import os
        import sys
        SECRET = "x"
        DEBUG = True
        """
    )
    chunks = chunk_file(sf, min_chunk_lines=10)

    assert len(chunks) == 1, "four trivial lines should not be four embeddings"
    assert chunks[0].num_lines == 4


def test_merging_respects_max_chunk_lines():
    sf = make_file("\n".join(f"CONST_{i} = {i}" for i in range(50)))
    chunks = chunk_file(sf, min_chunk_lines=100, max_chunk_lines=10)

    assert all(c.num_lines <= 10 for c in chunks)


# ---------------------------------------------------------------------------
# Naive baseline + metadata
# ---------------------------------------------------------------------------

def test_naive_strategy_ignores_syntax():
    sf = make_file("def a():\n" + "\n".join(f"    y{i} = {i}" for i in range(60)))

    naive = chunk_file(sf, strategy="naive", max_chunk_lines=20, overlap_lines=0)
    aware = chunk_file(sf, strategy="code_aware", max_chunk_lines=200)

    assert all(c.kind == "window" and c.symbol is None for c in naive)
    assert len(naive) > 1
    # Code-aware keeps the whole function as one named unit.
    assert len(aware) == 1 and aware[0].symbol == "a"


def test_naive_and_code_aware_both_cover_the_file():
    sf = make_file("def a():\n" + "\n".join(f"    y{i} = {i}" for i in range(60)))
    total = len(sf.content.splitlines())

    for strategy in ("naive", "code_aware"):
        chunks = chunk_file(sf, strategy=strategy, max_chunk_lines=20)
        covered: set[int] = set()
        for c in chunks:
            covered.update(range(c.start_line, c.end_line + 1))
        assert covered == set(range(1, total + 1)), f"{strategy} lost lines"


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_file(make_file("x = 1"), strategy="magic")


def test_citation_and_embedding_text():
    sf = make_file(
        """
        def login():
            return 1
        """,
        path="backend/auth.py",
    )
    chunk = by_symbol(chunk_file(sf, min_chunk_lines=1))["login"]

    assert chunk.citation == "backend/auth.py:1-2"

    text = chunk.embedding_text
    # The path and symbol must be in the embedded text, not just the metadata:
    # a question mentioning "auth" should match on the path alone.
    assert "backend/auth.py" in text
    assert "login" in text
    assert "def login():" in text


def test_chunk_files_preserves_file_order():
    files = [
        make_file("def a(): return 1", path="a.py"),
        make_file("def b(): return 2", path="b.py"),
    ]
    chunks = chunk_files(files, min_chunk_lines=1)
    assert [c.file_path for c in chunks] == ["a.py", "b.py"]


def test_empty_file_produces_no_chunks():
    assert chunk_file(make_file("")) == []


# ---------------------------------------------------------------------------
# Regression: lines that begin with '(' (found by indexing a real TS project)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "line",
    [
        "(async () => {",
        "(a, b) => a + b;",
        "() => {}",
        "(",
        "  (x) => x",
    ],
)
def test_lines_starting_with_a_paren_do_not_crash(line):
    """`split("(")[0].split()[0]` raised IndexError: there is no leading word."""
    from app.services.chunker import _first_token, _match_declaration

    assert _first_token(line) == ""
    # Must not raise for any language that consults the first token.
    for language in ("java", "javascript", "typescript"):
        _match_declaration(line, language, at_top_level=True)


def test_typescript_file_opening_with_an_iife_chunks_cleanly():
    sf = make_file(
        """
        (async () => {
          const res = await fetch("/api/health");
          console.log(await res.json());
        })();

        export function helper(x: number): number {
          return x * 2;
        }
        """,
        path="boot.ts",
        language="typescript",
    )

    chunks = chunk_file(sf, min_chunk_lines=1)

    assert chunks, "file produced no chunks"
    assert "helper" in by_symbol(chunks)
    covered: set[int] = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line + 1))
    lines = sf.content.splitlines()
    missing = [i + 1 for i, l in enumerate(lines) if l.strip() and (i + 1) not in covered]
    assert not missing, f"lines dropped: {missing}"
