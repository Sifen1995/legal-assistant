from typing import Dict

from .nodes import (
    QueryAnalysisNode,
    RetrievalNode,
    RerankerNode,
    SynthesisNode,
    WebSearchNode,
    GraderNode,
)
from .state import LegalDocument, LegalState
from ..engine.vector_store import VectorSearchTool
from ..engine.chunking_logic import build_chunks_from_pdf
from ..tools.search_tool import SearchTool


class LegalRAGWorkflow:
    def __init__(self, persist_directory: str = "chroma_store"):
        self.vector_store = VectorSearchTool(persist_directory=persist_directory)
        self.search_tool = SearchTool(self.vector_store)
        self.analysis_node = QueryAnalysisNode()
        self.retrieval_node = RetrievalNode(self.search_tool)
        self.reranker_node = RerankerNode()
        self.synthesis_node = SynthesisNode()
        self.web_search_node = WebSearchNode()
        self.grader_node = GraderNode()

    def run(self, initial_query: str) -> LegalState:
        state: LegalState = {
            "initial_query": initial_query,
            "search_plan": None,
            "retrieved_docs": [],
            "relevant_precedents": [],
            "legal_opinion": "",
            "citations": [],
            "grader_feedback": None,
        }

        state = self.analysis_node.execute(state)
        state = self.retrieval_node.execute(state)

        if state.get("search_plan") and state["search_plan"].needs_web_search:
            state = self.web_search_node.execute(state)
        else:
            state = self.reranker_node.execute(state)
            state = self.synthesis_node.execute(state)
            state = self.grader_node.execute(state)

            # If grading fails, regenerate synthesis with feedback (up to 1 retry)
            if state.get("grade") == "FAIL":
                state = self.synthesis_node.execute(state)
                state = self.grader_node.execute(state)  # Re-grade after regeneration

        return state

    def ingest_pdf(self, pdf_path: str) -> None:
        chunks = build_chunks_from_pdf(pdf_path)
        documents = [
            LegalDocument(
                content=chunk["content"],
                source=chunk["source"],
                page=chunk.get("page"),
                article_ref=chunk.get("article_ref"),
            )
            for chunk in chunks
        ]
        self.vector_store.add_documents(documents)

    def ingest_data_dir(self, folder: str = "data") -> None:
        from pathlib import Path

        folder_path = Path(folder)
        for pdf_path in sorted(folder_path.glob("*.pdf")):
            self.ingest_pdf(str(pdf_path))

    def load_documents(self, documents: list[LegalDocument]) -> None:
        self.vector_store.add_documents(documents)
