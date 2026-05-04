# Architecture: Agentic Legal RAG

## 1) System Components

- **API Layer (`app.py`)**
  - `GET /`: serves the built-in frontend (`static/index.html`)
  - `GET /health`: health endpoint
  - `POST /query`: executes the legal RAG workflow
- **Frontend (`static/`)**
  - Plain HTML/CSS/JS client that submits a query to `/query` and renders legal opinion + citations
- **Workflow Layer (`src/graph/`)**
  - Node-based legal reasoning and retrieval pipeline
- **Retrieval Layer (`src/engine/`, `src/tools/`)**
  - Chroma-backed vector search with metadata filters and chunk-based retrieval
- **Data Layer**
  - Input PDFs in `data/`
  - Persistent vector store in `chroma_store/` (or custom `persist_dir`)

## 2) Runtime Request Flow

### API + Frontend Flow

1. User opens `/` and submits a legal question.
2. Frontend sends `POST /query` with:
   - `query` (required)
   - `persist_dir` (optional, default `chroma_store`)
   - `ingest_path` (optional)
3. Backend initializes `LegalRAGWorkflow`.
4. Ingestion behavior:
   - If `ingest_path` is provided, ingest that PDF first.
   - Otherwise, ingest all PDFs in `data/` if present.
5. Backend runs the workflow and returns:
   - `legal_opinion`
   - `citations`
   - `needs_web_search`

### Workflow Flow (`LegalRAGWorkflow.run`)

```text
Initial Query
  -> QueryAnalysisNode
  -> RetrievalNode
  -> if search_plan.needs_web_search:
       WebSearchNode
     else:
       RerankerNode
       SynthesisNode
       GraderNode
       if grade == FAIL:
         SynthesisNode (retry once with grader feedback)
         GraderNode (re-check)
```

## 3) Node Responsibilities

- **QueryAnalysisNode**
  - Uses Groq (`llama3-8b-8192`) to build `search_plan`:
    - refined query
    - metadata filters (topic/year)
    - web-search flag
  - Falls back to heuristic extraction if LLM call fails.

- **RetrievalNode**
  - Retrieves matching documents through `SearchTool` / vector search.
  - If no docs are found, flips `search_plan.needs_web_search = True`.

- **RerankerNode**
  - Sorts docs by score descending.
  - Keeps docs with score >= `0.08`; otherwise falls back to top 3.

- **SynthesisNode**
  - Uses Gemini (`gemini-1.5-flash`) to produce legal opinion and citations.
  - Falls back to a deterministic text report if LLM generation fails.

- **GraderNode**
  - Uses Groq to evaluate hallucination/grounding quality.
  - Outputs `PASS`, `FAIL`, or `UNKNOWN`, plus optional feedback.
  - On `FAIL`, synthesis is retried once using grader feedback.

- **WebSearchNode**
  - Placeholder behavior for now.
  - Returns a message indicating the workflow would route to public web search.

## 4) Models and Keys

- **Groq** (`GROQ_API_KEY`):
  - Query analysis
  - Grading
- **Google Gemini** (`GMINI_API_KEY`):
  - Legal synthesis
- **Optional** `TAVILY_API_KEY`:
  - Reserved for future web search integration

## 5) Deployment Architecture (Render)

- `render.yaml` defines a Python web service:
  - Build: `pip install -r requirements.txt`
  - Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- GitHub Actions workflow (`.github/workflows/render-deploy.yml`) can trigger Render deploy via:
  - `RENDER_DEPLOY_HOOK_URL` (recommended), or
  - `RENDER_API_KEY` + `RENDER_SERVICE_ID`


