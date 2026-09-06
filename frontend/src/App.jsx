import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "./api.js";
import ChatPanel from "./components/ChatPanel.jsx";
import RepositorySidebar from "./components/RepositorySidebar.jsx";

// Indexing is a background job on the server, so the UI has to ask. Two
// seconds is frequent enough to feel live without hammering the API.
const POLL_INTERVAL_MS = 2000;

export default function App() {
  const [repositories, setRepositories] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [messagesByRepo, setMessagesByRepo] = useState({});
  const [conversationByRepo, setConversationByRepo] = useState({});
  const [thinking, setThinking] = useState(false);
  const [adding, setAdding] = useState(false);
  const [apiError, setApiError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const list = await api.listRepositories();
      setRepositories(list);
      setApiError(null);
      return list;
    } catch (error) {
      setApiError(error.message);
      return null;
    }
  }, []);

  // Loading the repository list is synchronisation with an external system,
  // which is exactly what effects are for. The lint rule cannot tell that
  // apart from a derived-state mistake.
  useEffect(() => {
    // eslint-disable-next-line react/set-state-in-effect
    refresh();
  }, [refresh]);

  // Poll only while something is actually being indexed. Polling forever would
  // keep a request in flight every two seconds for no reason.
  useEffect(() => {
    const busy = repositories.some((r) =>
      ["pending", "indexing"].includes(r.index_status)
    );
    if (!busy) return;

    const timer = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [repositories, refresh]);

  const selected = repositories.find((r) => r.id === selectedId) ?? null;
  const messages = messagesByRepo[selectedId] ?? [];

  const appendMessage = (repoId, message) =>
    setMessagesByRepo((current) => ({
      ...current,
      [repoId]: [...(current[repoId] ?? []), message],
    }));

  const handleAdd = async ({ mode, value, file, name }) => {
    setAdding(true);
    setApiError(null);
    try {
      const created =
        mode === "zip"
          ? await api.uploadRepository(file, name || undefined)
          : await api.registerRepository({
              [mode]: value,
              name: name || undefined,
            });

      const list = await refresh();
      // Select it straight away so the indexing progress is visible.
      if (list?.some((r) => r.id === created.id)) setSelectedId(created.id);
    } catch (error) {
      setApiError(error.message);
    } finally {
      setAdding(false);
    }
  };

  const handleDelete = async (id) => {
    try {
      await api.deleteRepository(id);
      if (selectedId === id) setSelectedId(null);
      setMessagesByRepo((current) => {
        const next = { ...current };
        delete next[id];
        return next;
      });
      await refresh();
    } catch (error) {
      setApiError(error.message);
    }
  };

  const handleReindex = async (id) => {
    try {
      await api.reindexRepository(id);
      await refresh();
    } catch (error) {
      setApiError(error.message);
    }
  };

  const handleAsk = async (question) => {
    if (!selected) return;
    const repoId = selected.id;

    appendMessage(repoId, { role: "user", content: question });
    setThinking(true);

    try {
      const response = await api.chat({
        repository_id: repoId,
        question,
        conversation_id: conversationByRepo[repoId] ?? undefined,
      });

      setConversationByRepo((current) => ({
        ...current,
        [repoId]: response.conversation_id,
      }));

      appendMessage(repoId, {
        role: "assistant",
        content: response.answer,
        citations: response.citations,
        sources: response.sources,
        is_grounded: response.is_grounded,
        no_context: response.no_context,
        citation_coverage: response.citation_coverage,
        invalid_markers: response.invalid_markers,
        model: response.model,
      });
    } catch (error) {
      appendMessage(repoId, {
        role: "error",
        content:
          error instanceof ApiError
            ? error.message
            : "Something went wrong asking that question.",
      });
    } finally {
      setThinking(false);
    }
  };

  return (
    <div className="flex h-full">
      <RepositorySidebar
        repositories={repositories}
        selectedId={selectedId}
        onSelect={setSelectedId}
        onAdd={handleAdd}
        onDelete={handleDelete}
        onReindex={handleReindex}
        adding={adding}
        apiError={apiError}
      />
      <ChatPanel
        repository={selected}
        messages={messages}
        onAsk={handleAsk}
        thinking={thinking}
      />
    </div>
  );
}
