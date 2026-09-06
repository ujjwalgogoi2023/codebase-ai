import { useState } from "react";

const KIND_COLORS = {
  function: "bg-sky-100 text-sky-700",
  method: "bg-violet-100 text-violet-700",
  class: "bg-amber-100 text-amber-700",
  module: "bg-slate-100 text-slate-600",
  section: "bg-emerald-100 text-emerald-700",
  window: "bg-slate-100 text-slate-600",
};

/**
 * One cited source, collapsed to a citation line until clicked.
 *
 * Spec section 13 asks that clicking a citation show the code. That matters
 * more than it sounds: a citation nobody can check is just a claim wearing a
 * file path. Being able to expand the exact lines the answer was built from is
 * what makes the grounding verifiable rather than decorative.
 */
export default function SourceCard({ source, index, expanded, onToggle }) {
  const [copied, setCopied] = useState(false);

  const copy = async (event) => {
    event.stopPropagation();
    try {
      await navigator.clipboard.writeText(source.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard unavailable (insecure origin); not worth interrupting for */
    }
  };

  const kindClass = KIND_COLORS[source.kind] ?? KIND_COLORS.module;

  return (
    <div
      id={`source-${index}`}
      className="overflow-hidden rounded-lg border border-slate-200 bg-white"
    >
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-slate-50"
      >
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-slate-800 font-mono text-[11px] text-white">
          {index}
        </span>

        <span className="truncate font-mono text-xs text-slate-700">
          {source.file_path}
          <span className="text-slate-400">
            :{source.start_line}-{source.end_line}
          </span>
        </span>

        {source.symbol && (
          <span className="hidden truncate font-mono text-xs text-slate-500 sm:inline">
            {source.symbol}
          </span>
        )}

        <span className={`ml-auto shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${kindClass}`}>
          {source.kind}
        </span>

        {typeof source.score === "number" && (
          <span
            className="shrink-0 font-mono text-[10px] text-slate-400"
            title="cosine similarity"
          >
            {source.score.toFixed(3)}
          </span>
        )}

        <span className="shrink-0 text-slate-400">{expanded ? "−" : "+"}</span>
      </button>

      {expanded && (
        <div className="relative border-t border-slate-200">
          <button
            onClick={copy}
            className="absolute top-2 right-2 z-10 rounded bg-slate-700 px-2 py-1 text-[10px] text-slate-200 hover:bg-slate-600"
          >
            {copied ? "copied" : "copy"}
          </button>

          {/* Line numbers come from the chunk's real start_line, so what is on
              screen matches what the citation claims. */}
          <pre className="overflow-x-auto bg-slate-900 p-3 text-xs leading-relaxed text-slate-100">
            <code className="font-mono">
              {source.content.split("\n").map((line, i) => (
                <div key={i} className="flex">
                  <span className="mr-3 w-10 shrink-0 select-none text-right text-slate-600">
                    {source.start_line + i}
                  </span>
                  <span className="whitespace-pre">{line}</span>
                </div>
              ))}
            </code>
          </pre>
        </div>
      )}
    </div>
  );
}
