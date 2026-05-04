from pathlib import Path
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


@app.get("/")
def serve_frontend() -> FileResponse:
    frontend_file = static_dir / "index.html"
    if not frontend_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(frontend_file)


@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest) -> QueryResponse:
    from src.graph.workflow import LegalRAGWorkflow

    workflow = LegalRAGWorkflow(persist_directory=request.persist_dir)

    if request.ingest_path:
        ingestion_path = Path(request.ingest_path)
        if not ingestion_path.exists():
            raise HTTPException(status_code=404, detail=f"Ingest path not found: {ingestion_path}")
        workflow.ingest_pdf(str(ingestion_path))
    else:
        data_folder = Path("data")
        pdf_files = sorted(data_folder.glob("*.pdf")) if data_folder.exists() else []
        if pdf_files:
            for pdf_path in pdf_files:
                workflow.ingest_pdf(str(pdf_path))

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
