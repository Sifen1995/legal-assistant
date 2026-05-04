from __future__ import annotations

import os
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

from ..graph.state import LegalDocument


class VectorSearchTool:
    def __init__(
        self,
        persist_directory: str = "chroma_store",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        self.collection = self.client.get_or_create_collection(
            name="legal_documents",
            embedding_function=self.embedding_fn,
        )

    def add_documents(self, documents: List[LegalDocument]) -> None:
        document_ids = [
            f"{document.source}-{index + 1}"
            for index, document in enumerate(documents)
        ]
        metadatas = [
            {
                "source": document.source,
                "page": document.page,
                "article_ref": document.article_ref,
            }
            for document in documents
        ]
        contents = [document.content for document in documents]

        if contents:
            try:
                self.collection.upsert(
                    ids=document_ids,
                    metadatas=metadatas,
                    documents=contents,
                )
            except Exception:
                self.collection.add(
                    ids=document_ids,
                    metadatas=metadatas,
                    documents=contents,
                )

    def search(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[dict] = None,
    ) -> List[LegalDocument]:
        if not query:
            return []

        query_results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        documents = query_results.get("documents", [[]])[0]
        metadatas = query_results.get("metadatas", [[]])[0]
        distances = query_results.get("distances", [[]])[0]

        for content, metadata, distance in zip(documents, metadatas, distances):
            doc = LegalDocument(
                content=content,
                source=metadata.get("source", "unknown"),
                page=metadata.get("page"),
                article_ref=metadata.get("article_ref"),
                score=1.0 / (1.0 + float(distance)) if distance is not None else 0.0,
            )
            hits.append(doc)

        if filters:
            hits = [hit for hit in hits if self._matches_filters(hit, filters)]

        return hits

    def _matches_filters(self, document: LegalDocument, filters: dict) -> bool:
        for key, value in filters.items():
            if key == "topic" and value not in document.content.lower():
                return False
            if key == "year" and (document.article_ref is None or value not in document.article_ref):
                return False
        return True
