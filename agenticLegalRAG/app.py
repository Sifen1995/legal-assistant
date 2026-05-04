from pathlib import Path
from functools import lru_cache
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class QueryRequest(BaseModel):
    query: str
    persist_dir: Optional[str] = "chroma_store"
    ingest_path: Optional[str] = None


class QueryResponse(BaseModel):
    legal_opinion: str
    citations: list[str]
    needs_web_search: bool


app = FastAPI(
    title="Agentic Legal RAG API",
    description="A FastAPI wrapper for the graph-based legal retrieval workflow.",
    version="0.1.0",
)

static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

_INGESTION_LOCK = Lock()
_INGESTED_SOURCES: set[str] = set()


@lru_cache(maxsize=4)
def _get_workflow(persist_dir: str):
    from src.graph.workflow import LegalRAGWorkflow

    return LegalRAGWorkflow(persist_directory=persist_dir)


def _ingest_pdf_once(workflow, persist_dir: str, pdf_path: Path) -> None:
    source_key = f"{persist_dir}:{pdf_path.resolve()}"
    with _INGESTION_LOCK:
        if source_key in _INGESTED_SOURCES:
            return
        workflow.ingest_pdf(str(pdf_path))
        _INGESTED_SOURCES.add(source_key)


@app.get("/")
def serve_frontend() -> FileResponse:
    frontend_file = static_dir / "index.html"
    if not frontend_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(frontend_file)


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    favicon_file = static_dir / "favicon.ico"
    if not favicon_file.exists():
        raise HTTPException(status_code=404, detail="Favicon not found.")
    return FileResponse(favicon_file)


@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest) -> QueryResponse:
    workflow = _get_workflow(request.persist_dir)

    if request.ingest_path:
        ingestion_path = Path(request.ingest_path)
        if not ingestion_path.exists():
            raise HTTPException(status_code=404, detail=f"Ingest path not found: {ingestion_path}")
        _ingest_pdf_once(workflow, request.persist_dir, ingestion_path)
    else:
        data_folder = Path("data")
        pdf_files = sorted(data_folder.glob("*.pdf")) if data_folder.exists() else []
        if pdf_files:
            for pdf_path in pdf_files:
                _ingest_pdf_once(workflow, request.persist_dir, pdf_path)

    result = workflow.run(request.query)
    search_plan = result.get("search_plan")
    needs_web_search = False
    if search_plan is not None:
        needs_web_search = getattr(search_plan, "needs_web_search", False)

    return QueryResponse(
        legal_opinion=result.get("legal_opinion", ""),
        citations=result.get("citations", []),
        needs_web_search=needs_web_search,
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "Agentic Legal RAG API is running."}
