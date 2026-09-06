import { useState } from "react";
import SourceCard from "./SourceCard.jsx";

/**
 * Split an answer so that [1] and [2,3] become clickable chips.
 *
 * Only markers matching a real citation become chips. The backend already
 * removed fabricated ones, but a leftover bracket in prose must still render
 * as plain text rather than a link to nothing.
 */
function renderWithCitations(text, validIndices, onCite) {
  const parts = text.split(/(\[\d+(?:\s*[,-]\s*\d+)*\])/g);

  return parts.map((part, i) => {
    const match = part.match(/^\[([\d\s,-]+)\]$/);
    if (!match) return <span key={i}>{part}</span>;

    const numbers = match[1]
      .split(",")
      .map((n) => parseInt(n.trim(), 10))
      .filter((n) => validIndices.has(n));

    if (numbers.length === 0) return <span key={i}>{part}</span>;

    return (
      <span key={i}>
        {numbers.map((n) => (
          <button
            key={n}
            onClick={() => onCite(n)}
            title="Show the cited code"
            className="mx-0.5 inline-flex h-4 w-4 items-center justify-center rounded bg-slate-800 align-text-top font-mono text-[10px] text-white hover:bg-sky-600"
          >
            {n}
          </button>
        ))}
      </span>
    );
  });
}

export default function MessageBubble({ message }) {
  const [expanded, setExpanded] = useState(() => new Set());
  const [showSources, setShowSources] = useState(false);

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-2xl rounded-2xl rounded-br-sm bg-slate-800 px-4 py-2.5 text-sm text-white">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
        {message.content}
      </div>
    );
  }

  const citations = message.citations ?? [];
  const validIndices = new Set(citations.map((c) => c.index));

  const toggle = (index) =>
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });

  const focusCitation = (index) => {
    setShowSources(true);
    setExpanded((current) => new Set(current).add(index));
    // Let the panel render before scrolling to it.
    requestAnimationFrame(() =>
      document
        .getElementById(`source-${index}`)
        ?.scrollIntoView({ behavior: "smooth", block: "center" })
    );
  };

  return (
    <div className="max-w-3xl space-y-3">
      <div className="rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3">
        <p className="text-sm leading-relaxed whitespace-pre-wrap text-slate-800">
          {renderWithCitations(message.content, validIndices, focusCitation)}
        </p>

        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-slate-100 pt-2 text-[11px]">
          {message.no_context ? (
            <span className="rounded bg-amber-100 px-1.5 py-0.5 font-medium text-amber-800">
              no matching code — the model was never called
            </span>
          ) : message.is_grounded ? (
            <span className="rounded bg-emerald-100 px-1.5 py-0.5 font-medium text-emerald-700">
              grounded
            </span>
          ) : (
            <span className="rounded bg-rose-100 px-1.5 py-0.5 font-medium text-rose-700">
              ungrounded
            </span>
          )}

          {typeof message.citation_coverage === "number" && !message.no_context && (
            <span className="text-slate-500" title="share of sentences carrying a citation">
              {Math.round(message.citation_coverage * 100)}% cited
            </span>
          )}

          {message.invalid_markers?.length > 0 && (
            <span
              className="rounded bg-rose-100 px-1.5 py-0.5 font-medium text-rose-700"
              title="The model cited sources that do not exist. They were removed."
            >
              {message.invalid_markers.length} fabricated citation
              {message.invalid_markers.length > 1 ? "s" : ""} removed
            </span>
          )}

          {message.model && (
            <span className="ml-auto font-mono text-slate-400">{message.model}</span>
          )}
        </div>
      </div>

      {citations.length > 0 && (
        <div className="space-y-1.5">
          <button
            onClick={() => setShowSources((v) => !v)}
            className="text-xs font-medium text-slate-500 hover:text-slate-800"
          >
            {showSources ? "▾" : "▸"} Sources ({citations.length})
          </button>

          {showSources && (
            <div className="space-y-1.5">
              {citations.map((citation) => {
                // The full chunk content lives on `sources`; citations carry
                // only the location. Join them so the code can be shown.
                const source =
                  message.sources?.find((s) => s.chunk_id === citation.chunk_id) ??
                  citation;
                return (
                  <SourceCard
                    key={citation.index}
                    index={citation.index}
                    source={{ ...citation, content: source.content ?? "" }}
                    expanded={expanded.has(citation.index)}
                    onToggle={() => toggle(citation.index)}
                  />
                );
              })}
            </div>
          )}
        </div>
      )}

      {citations.length === 0 && message.sources?.length > 0 && (
        <p className="text-xs text-slate-400">
          {message.sources.length} chunks were retrieved but the answer cited none
          of them.
        </p>
      )}
    </div>
  );
}
