"""Code-aware chunking: split source files into retrievable units.

Stage 2 of the RAG ingestion pipeline:

    Repository -> Parsing -> [CHUNKING] -> Embeddings -> pgvector

WHY THIS MATTERS
----------------
A chunk is the atomic unit of retrieval. Whatever a chunk contains is what the
LLM sees, and nothing smaller can ever be retrieved. So the chunk boundary
decides which questions the system can answer at all.

Naive chunking (`text[i:i+500]`) cuts wherever the character counter lands:

    def authenticate_user(username, password):
        user = db.get_user(username)
        if not user:
            return None
        if not bcrypt.checkpw(passw      <-- chunk boundary lands here

Now the embedding for chunk A is a function with no ending, and chunk B is a
fragment with no name. Neither embeds as "authenticating a user", so the query
"where do we authenticate users?" matches neither one well.

Code-aware chunking cuts on syntactic boundaries instead, so a function stays
whole and its embedding actually means what the function does.

THE TRADEOFF (spec section 6)
-----------------------------
  Too small -> loses surrounding context; a 2-line getter becomes its own
               embedding and competes for a top-K slot it does not deserve.
  Too large -> retrieval returns a 400-line class to answer a 3-line question,
               burning LLM context and diluting the signal.

We handle both ends explicitly: oversized units are windowed with overlap, and
trivially small adjacent units are merged.

Both strategies are implemented ("naive" and "code_aware") because the point is
to MEASURE the difference later (spec section 21), not to assert it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from app.services.repository_parser import SourceFile

# Defaults. Every one of these is an experiment knob for section 21.
DEFAULT_MAX_CHUNK_LINES = 120
DEFAULT_MIN_CHUNK_LINES = 5
DEFAULT_OVERLAP_LINES = 10

# Referenced-definition context. A definition longer than this is a table or a
# big literal, not a setting worth repeating next to every chunk that uses it.
MAX_DEFINITION_LINES = 6
MAX_ATTACHED_DEFINITIONS = 8


@dataclass(frozen=True)
class CodeChunk:
    """One retrievable unit of source code.

    Line numbers are 1-indexed and inclusive on both ends, matching how editors
    and GitHub display them, so a citation of "auth.py:42-61" points exactly
    where the user expects.
    """

    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol: str | None
    kind: str  # function | class | method | module | section | window
    # Module-level definitions this chunk refers to but does not contain, e.g.
    # `MAX_FAILED_ATTEMPTS = 5` for a login function that checks it. Shown to
    # the LLM next to the chunk; deliberately NOT part of content (it would
    # break the line range a citation points at) or of embedding_text.
    context: str | None = None

    @property
    def num_lines(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def embedding_text(self) -> str:
        """What we actually send to the embedding model.

        The raw code alone is missing information that the user's question
        contains. Someone asking "where is JWT validated in the auth
        middleware?" uses words like "auth" and "middleware" that may appear
        only in the FILE PATH, never in the function body. Prefixing the path
        and symbol puts those tokens into the vector, which helps retrieval.
        """
        header = f"File: {self.file_path}\nLanguage: {self.language}"
        if self.symbol:
            header += f"\nSymbol: {self.symbol}"
        return f"{header}\n\n{self.content}"


@dataclass
class _Span:
    """A region of source claimed by a unit, before post-processing.

    start/end are 1-indexed inclusive line numbers.
    """

    start: int
    end: int
    symbol: str | None
    kind: str


# ---------------------------------------------------------------------------
# Python: real AST parsing
# ---------------------------------------------------------------------------

def _python_spans(source: str, max_lines: int) -> list[_Span] | None:
    """Find functions/classes via the stdlib AST.

    Returns None if the file does not parse, so the caller can fall back to
    windowing rather than losing the file entirely. Real repositories contain
    Python 2 files, templates, and partially-broken code.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    spans: list[_Span] = []

    def start_of(node: ast.AST) -> int:
        # A decorator is part of the function for retrieval purposes:
        # @requires_auth sitting above a handler is exactly the kind of thing
        # someone searches for.
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            return min(d.lineno for d in decorators)
        return node.lineno

    def visit_class(node: ast.ClassDef, prefix: str = "") -> None:
        name = f"{prefix}{node.name}"
        start, end = start_of(node), node.end_lineno or node.lineno

        # Small class: keep it whole, the class IS the logical unit.
        if end - start + 1 <= max_lines:
            spans.append(_Span(start, end, name, "class"))
            return

        # Large class: emit each method separately so retrieval can land on the
        # specific method. The class header (docstring, class attributes) gets
        # picked up by gap-filling.
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append(
                    _Span(
                        start_of(child),
                        child.end_lineno or child.lineno,
                        f"{name}.{child.name}",
                        "method",
                    )
                )
            elif isinstance(child, ast.ClassDef):
                visit_class(child, prefix=f"{name}.")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append(
                _Span(start_of(node), node.end_lineno or node.lineno, node.name, "function")
            )
        elif isinstance(node, ast.ClassDef):
            visit_class(node)

    return spans


# ---------------------------------------------------------------------------
# JavaScript / TypeScript / Java: brace matching
# ---------------------------------------------------------------------------
# No stdlib parser exists for these, and pulling in tree-sitter would be a heavy
# dependency for stage 2. Brace matching handles the overwhelming majority of
# real declarations; anything it misses falls through to gap-filling, so no code
# is ever lost.

_JS_DECL = re.compile(
    r"""^\s*(?:export\s+)?(?:default\s+)?(?:
          (?:async\s+)?function\s*\*?\s*(?P<fn>[A-Za-z_$][\w$]*)
        | class\s+(?P<cls>[A-Za-z_$][\w$]*)
        | (?:const|let|var)\s+(?P<arrow>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*
              (?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>
        | (?:const|let|var)\s+(?P<fnexpr>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?function
    )""",
    re.VERBOSE,
)

# Methods inside a class body: `foo(a, b) {`, `async foo() {`, `get foo() {`.
_JS_METHOD = re.compile(
    r"^\s*(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?"
    r"(?P<name>[A-Za-z_$#][\w$]*)\s*\([^;]*\)\s*(?::\s*[^{;]+)?\{"
)

_JAVA_TYPE = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|private|protected|abstract|final|static|sealed|strictfp)\s+)*"
    r"(?:class|interface|enum|record)\s+(?P<name>[A-Za-z_$][\w$]*)"
)

# Modifiers are optional so package-private methods (`void run() {`) are still
# detected. That looseness is safe only because of the first-token guard in
# _match_declaration below, which rejects control-flow lines like
# `new Runnable() {` that would otherwise parse as "type name(...) {".
_JAVA_METHOD = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|private|protected|static|final|abstract|synchronized|native|default)\s+)*"
    r"(?:<[^>]+>\s*)?"
    r"[\w$<>\[\],.?]+(?:\s*\[\])?\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*\([^;]*\)\s*(?:throws\s+[\w$.,\s]+)?\{"
)

# Words that open a control block rather than a declaration. Without this guard
# `new Runnable() {` would be read as a method named `Runnable`.
_CONTROL_KEYWORDS = frozenset(
    {"if", "for", "while", "switch", "catch", "try", "do", "else", "return", "new"}
)


def _mask_line(line: str, in_block_comment: bool) -> tuple[str, bool]:
    """Blank out string and comment content so brace counting is not fooled.

    A line like `const s = "}}}"` must not look like three closing braces.
    Characters are replaced with spaces rather than deleted, so the regexes
    above still see correct column positions.
    """
    out: list[str] = []
    i, n = 0, len(line)
    quote: str | None = None

    while i < n:
        ch = line[i]

        if in_block_comment:
            if ch == "*" and i + 1 < n and line[i + 1] == "/":
                in_block_comment = False
                out.append("  ")
                i += 2
                continue
            out.append(" ")
            i += 1
            continue

        if quote is not None:
            if ch == "\\":
                out.append("  ")
                i += 2
                continue
            if ch == quote:
                quote = None
            out.append(" ")
            i += 1
            continue

        if ch == '"' or ch == "'" or ch == "`":
            quote = ch
            out.append(" ")
            i += 1
            continue

        if ch == "/" and i + 1 < n:
            if line[i + 1] == "/":
                out.append(" " * (n - i))
                break
            if line[i + 1] == "*":
                in_block_comment = True
                out.append("  ")
                i += 2
                continue

        out.append(ch)
        i += 1

    return "".join(out), in_block_comment


def _brace_depths(lines: list[str]) -> tuple[list[int], list[str]]:
    """Brace depth at the START of each line, plus the masked lines."""
    depths: list[int] = []
    masked: list[str] = []
    depth = 0
    in_block = False

    for line in lines:
        clean, in_block = _mask_line(line, in_block)
        depths.append(depth)
        masked.append(clean)
        depth += clean.count("{") - clean.count("}")

    return depths, masked


def _first_token(line: str) -> str:
    """The first word before any '(' on the line, or "" if there isn't one.

    Returning "" rather than indexing blindly matters: a line can legitimately
    begin with '(' -- an IIFE, or an arrow function whose parameter list starts
    the line -- in which case there is no leading word at all. An earlier
    version did `split("(")[0].split()[0]` and raised IndexError on the first
    TypeScript file that opened a line with a paren.
    """
    head = line.split("(")[0].split()
    return head[0] if head else ""


def _match_declaration(line: str, language: str, at_top_level: bool) -> tuple[str, str] | None:
    """Return (symbol, kind) if this line opens a declaration we care about."""
    if language == "java":
        m = _JAVA_TYPE.match(line)
        if m:
            return m.group("name"), "class"
        m = _JAVA_METHOD.match(line)
        if m and _first_token(line) not in _CONTROL_KEYWORDS:
            return m.group("name"), "method"
        return None

    # javascript / typescript
    m = _JS_DECL.match(line)
    if m:
        if m.group("cls"):
            return m.group("cls"), "class"
        name = m.group("fn") or m.group("arrow") or m.group("fnexpr")
        return name, "function"

    if not at_top_level:
        m = _JS_METHOD.match(line)
        if m and m.group("name") not in _CONTROL_KEYWORDS:
            return m.group("name"), "method"

    return None


def _brace_spans(lines: list[str], language: str, max_lines: int) -> list[_Span]:
    """Find declarations by scanning for matched brace blocks."""
    depths, masked = _brace_depths(lines)
    spans: list[_Span] = []

    def scan(lo: int, hi: int, depth: int, prefix: str) -> None:
        """Scan lines [lo, hi) for declarations sitting at `depth`."""
        i = lo
        while i < hi:
            if depths[i] != depth:
                i += 1
                continue

            decl = _match_declaration(masked[i], language, at_top_level=(depth == 0))
            if decl is None:
                i += 1
                continue

            symbol, kind = decl

            # Walk forward to the line where this block closes.
            end = i
            opened = False
            j = i
            while j < hi:
                opens = masked[j].count("{")
                closes = masked[j].count("}")
                if opens:
                    opened = True
                if opened and depths[j] + opens - closes <= depth:
                    end = j
                    break
                j += 1
            else:
                end = hi - 1

            if not opened:
                # A declaration with no body on this line (interface member,
                # abstract method). Not a useful standalone chunk.
                i += 1
                continue

            full_name = f"{prefix}{symbol}"
            size = end - i + 1

            if kind == "class" and size > max_lines:
                # Too big to keep whole: descend and chunk the methods instead.
                scan(i + 1, end, depth + 1, prefix=f"{full_name}.")
            else:
                spans.append(_Span(i + 1, end + 1, full_name, kind))

            i = end + 1

    scan(0, len(lines), 0, prefix="")
    return spans


# ---------------------------------------------------------------------------
# Markdown: split on headings
# ---------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*#*\s*$")


def _markdown_spans(lines: list[str]) -> list[_Span]:
    """Each heading starts a section that runs until the next heading.

    Docs answer real questions ("how does the payment flow work"), and a
    heading is the natural semantic boundary of a prose section.
    """
    headings = [
        (i, m.group("title"))
        for i, line in enumerate(lines)
        if (m := _MD_HEADING.match(line))
    ]
    if not headings:
        return []

    spans: list[_Span] = []
    for pos, (idx, title) in enumerate(headings):
        end = headings[pos + 1][0] - 1 if pos + 1 < len(headings) else len(lines) - 1
        spans.append(_Span(idx + 1, end + 1, title, "section"))
    return spans


# ---------------------------------------------------------------------------
# Shared post-processing: fill gaps -> merge tiny -> split oversized
# ---------------------------------------------------------------------------

def _fill_gaps(spans: list[_Span], total_lines: int) -> list[_Span]:
    """Claim every line not covered by a syntactic unit.

    Imports, module constants, license headers and the code between functions
    are all legitimately retrievable. "Where is the database connection
    initialized?" is usually answered by a module-level constant, not by a
    function. Gap filling guarantees the chunker never silently drops source.
    """
    spans = sorted(spans, key=lambda s: (s.start, s.end))

    # Drop spans nested inside an earlier span; the outer one already covers it.
    deduped: list[_Span] = []
    for span in spans:
        if deduped and span.start <= deduped[-1].end:
            continue
        deduped.append(span)

    filled: list[_Span] = []
    cursor = 1
    for span in deduped:
        if span.start > cursor:
            filled.append(_Span(cursor, span.start - 1, None, "module"))
        filled.append(span)
        cursor = span.end + 1

    if cursor <= total_lines:
        filled.append(_Span(cursor, total_lines, None, "module"))

    return filled


def _split_oversized(span: _Span, max_lines: int, overlap: int) -> list[_Span]:
    """Window a too-large unit, with overlap so we do not cut mid-thought.

    Overlap is insurance: if a boundary lands in the middle of a logical step,
    the following window repeats the tail, so the context still appears intact
    in at least one chunk.
    """
    size = span.end - span.start + 1
    if size <= max_lines:
        return [span]

    step = max(1, max_lines - overlap)
    out: list[_Span] = []
    start = span.start
    while start <= span.end:
        end = min(start + max_lines - 1, span.end)
        out.append(_Span(start, end, span.symbol, span.kind))
        if end == span.end:
            break
        start += step
    return out


def _merge_tiny(spans: list[_Span], min_lines: int, max_lines: int) -> list[_Span]:
    """Merge adjacent trivially-small units.

    A 2-line `__init__` or a 3-line import block is not worth its own
    embedding: it carries almost no semantic content, yet it occupies a top-K
    slot. Merging adjacent small units gives the embedding something to
    actually represent.
    """
    if not spans:
        return spans

    merged: list[_Span] = [spans[0]]
    for span in spans[1:]:
        prev = merged[-1]
        prev_size = prev.end - prev.start + 1
        span_size = span.end - span.start + 1
        combined = span.end - prev.start + 1

        contiguous = span.start == prev.end + 1
        both_small = prev_size < min_lines and span_size < min_lines

        if contiguous and both_small and combined <= max_lines:
            symbols = [s for s in (prev.symbol, span.symbol) if s]
            merged[-1] = _Span(
                prev.start,
                span.end,
                ", ".join(symbols) if symbols else None,
                prev.kind if prev.kind == span.kind else "module",
            )
        else:
            merged.append(span)

    return merged


def _naive_spans(total_lines: int, max_lines: int, overlap: int) -> list[_Span]:
    """The baseline we measure against: fixed windows, no syntax awareness."""
    step = max(1, max_lines - overlap)
    spans: list[_Span] = []
    start = 1
    while start <= total_lines:
        end = min(start + max_lines - 1, total_lines)
        spans.append(_Span(start, end, None, "window"))
        if end == total_lines:
            break
        start += step
    return spans


# ---------------------------------------------------------------------------
# Referenced definitions (Python)
# ---------------------------------------------------------------------------
# Code-aware chunking separates a function from the constants it reads: the
# login function is one chunk, `MAX_FAILED_ATTEMPTS = 5` is part of a module
# chunk near the top of the file. Retrieval finds the function, the LLM sees
# `if failed >= MAX_FAILED_ATTEMPTS`, and the actual number never reaches it.
# Measured on a real repository, that was the most common reason an answer
# found the right code but left out the value the question asked for.

def _python_module_definitions(source: str) -> dict[str, tuple[int, int]]:
    """Map each short module-level assignment or helper function to its lines.

    Tiny helpers count too: `_new_otp()` is two lines whose body IS the answer
    to "how is the code generated?", and it lives in a different chunk from the
    function that calls it.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}

    definitions: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        start = node.lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [node.name]
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(
            node.target, ast.Name
        ):
            names = [node.target.id]
        else:
            continue

        end = node.end_lineno or node.lineno
        if end - start + 1 > MAX_DEFINITION_LINES:
            continue
        for name in names:
            if not name.startswith("__"):  # __all__, __version__: not settings
                definitions[name] = (start, end)
    return definitions


def _referenced_definitions(
    span: _Span,
    content: str,
    lines: list[str],
    definitions: dict[str, tuple[int, int]],
) -> str | None:
    """Source of the module-level definitions `content` uses but lacks."""
    wanted: set[tuple[int, int]] = set()
    for name, (start, end) in definitions.items():
        if span.start <= start and end <= span.end:
            continue  # the chunk already contains this definition
        if re.search(rf"\b{re.escape(name)}\b", content):
            wanted.add((start, end))

    if not wanted:
        return None
    picked = sorted(wanted)[:MAX_ATTACHED_DEFINITIONS]
    return "\n".join("\n".join(lines[start - 1 : end]) for start, end in picked)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _semantic_spans(source_file: SourceFile, lines: list[str], max_lines: int) -> list[_Span]:
    """Dispatch to the right language strategy, falling back to windowing."""
    language = source_file.language

    if language == "python":
        spans = _python_spans(source_file.content, max_lines)
        if spans is None:
            # Unparseable Python still deserves to be searchable.
            return _naive_spans(len(lines), max_lines, DEFAULT_OVERLAP_LINES)
        return spans

    if language in ("javascript", "typescript", "java"):
        return _brace_spans(lines, language, max_lines)

    if language == "markdown":
        return _markdown_spans(lines)

    return []


def chunk_file(
    source_file: SourceFile,
    *,
    strategy: str = "code_aware",
    max_chunk_lines: int = DEFAULT_MAX_CHUNK_LINES,
    min_chunk_lines: int = DEFAULT_MIN_CHUNK_LINES,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> list[CodeChunk]:
    """Split one file into retrievable chunks.

    `strategy` is "code_aware" or "naive". Keeping both in the codebase is what
    makes the section 21 experiment possible: same corpus, same embeddings,
    same retriever, exactly one variable changed.
    """
    if strategy not in ("code_aware", "naive"):
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    lines = source_file.content.splitlines()
    if not lines:
        return []

    if strategy == "naive":
        spans = _naive_spans(len(lines), max_chunk_lines, overlap_lines)
    else:
        spans = _semantic_spans(source_file, lines, max_chunk_lines)
        spans = _fill_gaps(spans, len(lines))
        spans = _merge_tiny(spans, min_chunk_lines, max_chunk_lines)

    final: list[_Span] = []
    for span in spans:
        final.extend(_split_oversized(span, max_chunk_lines, overlap_lines))

    definitions = (
        _python_module_definitions(source_file.content)
        if strategy == "code_aware" and source_file.language == "python"
        else {}
    )

    chunks: list[CodeChunk] = []
    for span in final:
        content = "\n".join(lines[span.start - 1 : span.end])
        if not content.strip():
            continue  # a run of blank lines is not worth an embedding
        chunks.append(
            CodeChunk(
                content=content,
                file_path=source_file.path,
                language=source_file.language,
                start_line=span.start,
                end_line=span.end,
                symbol=span.symbol,
                kind=span.kind,
                context=_referenced_definitions(span, content, lines, definitions)
                if definitions
                else None,
            )
        )
    return chunks


def chunk_files(source_files: list[SourceFile], **kwargs) -> list[CodeChunk]:
    """Chunk many files, preserving file order."""
    chunks: list[CodeChunk] = []
    for source_file in source_files:
        chunks.extend(chunk_file(source_file, **kwargs))
    return chunks
