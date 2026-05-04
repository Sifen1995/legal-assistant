const form = document.getElementById("query-form");
const statusEl = document.getElementById("status");
const opinionEl = document.getElementById("legal-opinion");
const citationsEl = document.getElementById("citations");

function setStatus(message) {
  statusEl.textContent = message;
}

function renderCitations(citations) {
  citationsEl.innerHTML = "";
  if (!citations || citations.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No citations provided.";
    citationsEl.appendChild(li);
    return;
  }

  citations.forEach((citation) => {
    const li = document.createElement("li");
    li.textContent = citation;
    citationsEl.appendChild(li);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Running query...");
  opinionEl.textContent = "";
  citationsEl.innerHTML = "";

  const query = document.getElementById("query").value.trim();
  const persistDir = document.getElementById("persistDir").value.trim() || "chroma_store";
  const ingestPath = document.getElementById("ingestPath").value.trim();

  const body = {
    query,
    persist_dir: persistDir
  };

  if (ingestPath) {
    body.ingest_path = ingestPath;
  }

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const errBody = await response.text();
      throw new Error(errBody || `Request failed (${response.status})`);
    }

    const data = await response.json();
    opinionEl.textContent = data.legal_opinion || "No legal opinion returned.";
    renderCitations(data.citations || []);
    setStatus(data.needs_web_search ? "Done. Web search is recommended." : "Done.");
  } catch (error) {
    opinionEl.textContent = "An error occurred while running the query.";
    renderCitations([]);
    setStatus(`Error: ${error.message}`);
  }
});
