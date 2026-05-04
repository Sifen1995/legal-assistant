from typing import Dict, List, Optional

from ..engine.vector_store import VectorSearchTool
from ..graph.state import LegalDocument


class SearchTool:
    def __init__(self, vector_search_tool: VectorSearchTool):
        self.vector_search_tool = vector_search_tool

    def retrieve(self, query: str, filters: Optional[Dict[str, str]] = None) -> List[LegalDocument]:
        documents = self.vector_search_tool.search(query)
        if not filters:
            return documents

        return [document for document in documents if self._matches(document, filters)]

    def _matches(self, document: LegalDocument, filters: Dict[str, str]) -> bool:
        if "topic" in filters:
            topic = filters["topic"].lower()
            if topic not in document.content.lower():
                return False

        if "year" in filters and document.article_ref is not None:
            if filters["year"] not in document.article_ref:
                return False

        return True
