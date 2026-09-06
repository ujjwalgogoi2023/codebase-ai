import { useEffect, useRef, useState } from "react";
import MessageBubble from "./MessageBubble.jsx";

const EXAMPLES = [
  "Where is authentication implemented?",
  "How does the indexing pipeline work?",
  "Which function creates an order?",
  "Where is the database connection initialized?",
];

export default function ChatPanel({ repository, messages, onAsk, thinking }) {
  const [question, setQuestion] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, thinking]);

  const submit = (event) => {
    event.preventDefault();
    const text = question.trim();
    if (!text || thinking) return;
    onAsk(text);
    setQuestion("");
  };

  if (!repository) {
    return (
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="max-w-sm text-center">
          <h2 className="text-sm font-medium text-slate-700">
            Select a repository
          </h2>
          <p className="mt-1 text-xs text-slate-500">
            Add a codebase from the sidebar, wait for indexing to finish, then
            ask questions about it.
          </p>
        </div>
      </div>
    );
  }

  const notReady = repository.index_status !== "ready";

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <header className="flex items-center gap-3 border-b border-slate-200 bg-white px-5 py-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold text-slate-900">
            {repository.name}
          </h2>
          <p className="text-[11px] text-slate-500">
            {repository.file_count} files · {repository.chunk_count} chunks ·{" "}
            <span className="font-mono">{repository.embedding_model}</span>
          </p>
        </div>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
        {messages.length === 0 && !notReady && (
          <div className="mx-auto max-w-lg pt-8">
            <p className="mb-3 text-center text-xs text-slate-500">
              Ask anything about this codebase.
            </p>
            <div className="grid gap-2">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  onClick={() => onAsk(example)}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-xs text-slate-600 hover:border-slate-400 hover:text-slate-900"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {notReady && (
          <div className="mx-auto max-w-md rounded-lg border border-slate-200 bg-white p-4 text-center">
            <p className="text-sm font-medium text-slate-700">
              {repository.index_status === "failed"
                ? "Indexing failed"
                : "Indexing in progress"}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {repository.error_message ||
                "Parsing, chunking and embedding the repository. This page polls for updates."}
            </p>
          </div>
        )}

        {messages.map((message, i) => (
          <MessageBubble key={i} message={message} />
        ))}

        {thinking && (
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" />
            <span>Retrieving code and generating an answer…</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={submit}
        className="border-t border-slate-200 bg-white px-5 py-3"
      >
        <div className="flex gap-2">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={notReady || thinking}
            placeholder={
              notReady
                ? "Waiting for indexing to finish…"
                : "Ask about this codebase…"
            }
            className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-800 disabled:bg-slate-50"
          />
          <button
            type="submit"
            disabled={notReady || thinking || !question.trim()}
            className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-40"
          >
            Ask
          </button>
        </div>
        <p className="mt-1.5 text-[10px] text-slate-400">
          Every question runs a fresh search. Conversation history helps resolve
          references, but never replaces retrieval.
        </p>
      </form>
    </div>
  );
}
