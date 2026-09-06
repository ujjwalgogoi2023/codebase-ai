/**
 * Thin client for the CodebaseAI backend.
 *
 * Talks to the API cross-origin, which is why main.py configures CORS for
 * this dev-server port. A Vite proxy would hide that entirely; keeping the
 * real cross-origin call means the CORS configuration is exercised in
 * development rather than discovered in production.
 */

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: options.body instanceof FormData
        ? undefined
        : { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    // fetch only rejects on network-level failure, which in practice means
    // the backend is not running. Say that, rather than "Failed to fetch".
    throw new ApiError(
      `Cannot reach the API at ${API_BASE}. Is the backend running?`,
      0
    );
  }

  if (response.status === 204) return null;

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    // FastAPI puts a string in `detail` for HTTPException, and a list of
    // field errors there for validation failures.
    const detail = body?.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => `${d.loc?.at(-1)}: ${d.msg}`).join(", ")
      : detail || `Request failed (${response.status})`;
    throw new ApiError(message, response.status);
  }

  return body;
}

export const api = {
  health: () => request("/health"),

  listRepositories: () => request("/repositories"),

  getRepositoryStatus: (id) => request(`/repositories/${id}/status`),

  registerRepository: (payload) =>
    request("/repositories", { method: "POST", body: JSON.stringify(payload) }),

  uploadRepository: (file, name) => {
    const form = new FormData();
    form.append("archive", file);
    if (name) form.append("name", name);
    return request("/repositories/upload", { method: "POST", body: form });
  },

  reindexRepository: (id) =>
    request(`/repositories/${id}/index`, { method: "POST" }),

  deleteRepository: (id) =>
    request(`/repositories/${id}`, { method: "DELETE" }),

  listFiles: (id) => request(`/repositories/${id}/files`),

  search: (id, query, params = {}) => {
    const search = new URLSearchParams({ q: query, ...params });
    return request(`/repositories/${id}/search?${search}`);
  },

  chat: (payload) =>
    request("/chat", { method: "POST", body: JSON.stringify(payload) }),

  getConversation: (id) => request(`/conversations/${id}`),
};

export { ApiError, API_BASE };
