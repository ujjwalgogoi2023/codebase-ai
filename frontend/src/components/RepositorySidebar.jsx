import { useState } from "react";

const STATUS_STYLES = {
  ready: "bg-emerald-100 text-emerald-700",
  indexing: "bg-sky-100 text-sky-700",
  pending: "bg-slate-200 text-slate-600",
  failed: "bg-rose-100 text-rose-700",
};

function StatusBadge({ status }) {
  const busy = status === "indexing" || status === "pending";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${
        STATUS_STYLES[status] ?? STATUS_STYLES.pending
      }`}
    >
      {busy && (
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
      )}
      {status}
    </span>
  );
}

function AddRepositoryForm({ onSubmit, busy }) {
  const [mode, setMode] = useState("path");
  const [value, setValue] = useState("");
  const [file, setFile] = useState(null);
  const [name, setName] = useState("");

  const submit = (event) => {
    event.preventDefault();
    if (mode === "zip") {
      if (file) onSubmit({ mode, file, name });
    } else if (value.trim()) {
      onSubmit({ mode, value: value.trim(), name });
    }
    setValue("");
    setFile(null);
    setName("");
  };

  const tabs = [
    ["path", "Local"],
    ["url", "GitHub"],
    ["zip", "ZIP"],
  ];

  return (
    <form onSubmit={submit} className="space-y-2 border-b border-slate-200 p-3">
      <div className="flex gap-1">
        {tabs.map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => setMode(key)}
            className={`flex-1 rounded px-2 py-1 text-xs font-medium ${
              mode === key
                ? "bg-slate-800 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {mode === "zip" ? (
        <input
          type="file"
          accept=".zip"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="w-full text-xs file:mr-2 file:rounded file:border-0 file:bg-slate-200 file:px-2 file:py-1 file:text-xs"
        />
      ) : (
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={
            mode === "path"
              ? "C:\\path\\to\\repo"
              : "https://github.com/owner/repo"
          }
          className="w-full rounded border border-slate-300 px-2 py-1.5 font-mono text-xs outline-none focus:border-slate-800"
        />
      )}

      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="name (optional)"
        className="w-full rounded border border-slate-300 px-2 py-1.5 text-xs outline-none focus:border-slate-800"
      />

      <button
        type="submit"
        disabled={busy || (mode === "zip" ? !file : !value.trim())}
        className="w-full rounded bg-slate-800 px-2 py-1.5 text-xs font-medium text-white hover:bg-slate-700 disabled:opacity-40"
      >
        {busy ? "Adding…" : "Add repository"}
      </button>
    </form>
  );
}

export default function RepositorySidebar({
  repositories,
  selectedId,
  onSelect,
  onAdd,
  onDelete,
  onReindex,
  adding,
  apiError,
}) {
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-slate-200 bg-white">
      <header className="border-b border-slate-200 px-4 py-3">
        <h1 className="text-sm font-semibold text-slate-900">CodebaseAI</h1>
        <p className="text-[11px] text-slate-500">
          Ask questions about a codebase
        </p>
      </header>

      <AddRepositoryForm onSubmit={onAdd} busy={adding} />

      {apiError && (
        <div className="border-b border-rose-200 bg-rose-50 px-3 py-2 text-[11px] text-rose-700">
          {apiError}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto">
        {repositories.length === 0 ? (
          <p className="p-4 text-xs text-slate-400">
            No repositories yet. Add one above to get started.
          </p>
        ) : (
          <ul>
            {repositories.map((repo) => (
              <li
                key={repo.id}
                className={`group border-b border-slate-100 ${
                  selectedId === repo.id ? "bg-slate-100" : "hover:bg-slate-50"
                }`}
              >
                {/* A real <button>, not a clickable div: this is the primary
                    control in the sidebar, so it has to be reachable by
                    keyboard and announced as a button. The row actions are
                    siblings rather than children, because nesting a button
                    inside a button is invalid HTML. */}
                <button
                  onClick={() => onSelect(repo.id)}
                  aria-label={`Select repository ${repo.name}`}
                  aria-current={selectedId === repo.id ? "true" : undefined}
                  className="w-full px-3 pt-2.5 text-left"
                >
                  <span className="flex items-center gap-2">
                    <span className="truncate text-xs font-medium text-slate-800">
                      {repo.name}
                    </span>
                    <span className="ml-auto shrink-0">
                      <StatusBadge status={repo.index_status} />
                    </span>
                  </span>

                  <span className="mt-1 flex items-center gap-2 text-[10px] text-slate-500">
                    <span>{repo.file_count} files</span>
                    <span>·</span>
                    <span>{repo.chunk_count} chunks</span>
                    <span>·</span>
                    <span className="font-mono">{repo.chunk_strategy}</span>
                  </span>

                  {repo.error_message && (
                    <span className="mt-1 line-clamp-2 block text-[10px] text-rose-600">
                      {repo.error_message}
                    </span>
                  )}
                </button>

                <div className="flex gap-2 px-3 pt-1 pb-2.5 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                  <button
                    onClick={() => onReindex(repo.id)}
                    aria-label={`Re-index ${repo.name}`}
                    className="text-[10px] text-slate-500 hover:text-slate-900"
                  >
                    re-index
                  </button>
                  <button
                    onClick={() => onDelete(repo.id)}
                    aria-label={`Delete ${repo.name}`}
                    className="text-[10px] text-rose-500 hover:text-rose-700"
                  >
                    delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
