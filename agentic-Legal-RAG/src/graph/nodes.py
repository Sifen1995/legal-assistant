from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List
import os

from groq import Groq
from google import genai

from .state import LegalDocument, LegalState, QueryPlan
from ..tools.search_tool import SearchTool


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def _extract_metadata_filters(query: str) -> Dict[str, str]:
    filters: Dict[str, str] = {}
    lower_query = query.lower()

    for topic in ["housing", "contract", "employment", "tax", "fraud", "tenant", "landlord"]:
        if topic in lower_query:
            filters["topic"] = topic
            break

    tokens = [token.strip(".,") for token in lower_query.split()]
    for token in tokens:
        if token.isdigit() and len(token) == 4:
            filters["year"] = token
            break

    return filters


class Node(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, state: LegalState) -> LegalState:
        ...


class QueryAnalysisNode(Node):
    def __init__(self):
        super().__init__("QueryAnalysisNode")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def execute(self, state: LegalState) -> LegalState:
        raw_query = state.get("initial_query", "").strip()
        if not raw_query:
            raise ValueError("initial_query is required for QueryAnalysisNode")

        # Use Groq to analyze the query and extract metadata
        prompt = f"""
        Analyze this legal query and extract key information for search planning:

        Query: {raw_query}

        Please provide:
        1. Refined query (cleaned up version)
        2. Topic filters (comma-separated if multiple)
        3. Year filters (if mentioned)
        4. Whether web search might be needed

        Format your response as JSON:
        {{
            "refined_query": "string",
            "topics": "comma,separated,topics",
            "year": "YYYY or null",
            "needs_web_search": true/false
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )

            result = response.choices[0].message.content.strip()
            # Parse the JSON response (simplified parsing)
            import json
            analysis = json.loads(result)

            metadata_filters = {}
            if analysis.get("topics"):
                metadata_filters["topic"] = analysis["topics"].split(",")[0].strip()
            if analysis.get("year") and analysis["year"] != "null":
                metadata_filters["year"] = analysis["year"]

            state["search_plan"] = QueryPlan(
                refined_query=analysis.get("refined_query", raw_query),
                metadata_filters=metadata_filters,
                needs_web_search=analysis.get("needs_web_search", False),
            )
        except Exception as e:
            # Fallback to simple processing if LLM fails
            print(f"LLM analysis failed: {e}, using fallback")
            state["search_plan"] = QueryPlan(
                refined_query=_normalize_query(raw_query),
                metadata_filters=_extract_metadata_filters(raw_query),
                needs_web_search=False,
            )

        return state


class RetrievalNode(Node):
    def __init__(self, search_tool: SearchTool):
        super().__init__("RetrievalNode")
        self.search_tool = search_tool

    def execute(self, state: LegalState) -> LegalState:
        search_plan = state.get("search_plan")
        if not search_plan:
            raise ValueError("search_plan is required before retrieval")

        documents = self.search_tool.retrieve(
            search_plan.refined_query,
            filters=search_plan.metadata_filters,
        )

        if not documents:
            search_plan.needs_web_search = True

        state["retrieved_docs"] = documents
        state["search_plan"] = search_plan
        return state


class RerankerNode(Node):
    def __init__(self):
        super().__init__("RerankerNode")

    def execute(self, state: LegalState) -> LegalState:
        retrieved_docs = state.get("retrieved_docs", [])
        if not retrieved_docs:
            state["relevant_precedents"] = []
            return state

        ranked = sorted(retrieved_docs, key=lambda document: document.score, reverse=True)
        relevant = [doc for doc in ranked if doc.score >= 0.08]
        if not relevant:
            relevant = ranked[:3]

        state["relevant_precedents"] = relevant
        return state


class SynthesisNode(Node):
    def __init__(self):
        super().__init__("SynthesisNode")
        self.client = genai.Client(api_key=os.getenv("GMINI_API_KEY"))

    def execute(self, state: LegalState) -> LegalState:
        relevant_docs = state.get("relevant_precedents", [])
        if not relevant_docs:
            state["legal_opinion"] = (
                "No relevant precedents were found in the legal database. "
                "Please review the query or ingest more documents."
            )
            state["citations"] = []
            return state

        # Prepare context from relevant documents
        context_parts = []
        citations: List[str] = []

        for index, document in enumerate(relevant_docs[:3], start=1):
            citation = document.article_ref or f"{document.source}-page-{document.page}"
            citations.append(citation)
            context_parts.append(f"Document {index} ({citation}): {document.content}")

        context = "\n\n".join(context_parts)
        original_query = state.get("initial_query", "")
        grader_feedback = state.get("grader_feedback")

        prompt = f"""
        You are a legal expert. Based on the following legal documents, provide a comprehensive answer to the user's question.

        Original Question: {original_query}

        Relevant Legal Documents:
        {context}
        """

        if grader_feedback:
            prompt += f"\n\nPrevious Grading Feedback (Address these issues in your response):\n{grader_feedback}\n"

        prompt += """
        Please provide:
        1. A clear, concise answer to the question
        2. Key legal principles or rules from the documents
        3. Any important caveats or limitations
        4. Citations to the specific documents used

        Structure your response professionally as a legal opinion.
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=genai.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                )
            )

            state["legal_opinion"] = response.text
            state["citations"] = citations
        except Exception as e:
            # Fallback to simple synthesis if LLM fails
            print(f"LLM synthesis failed: {e}, using fallback")
            report_lines: List[str] = [f"Legal Opinion on: {original_query}\n"]
            report_lines.append("Based on the following relevant precedents:")

            for index, document in enumerate(relevant_docs[:3], start=1):
                citation = document.article_ref or document.source
                citations.append(citation)
                report_lines.append(
                    f"{index}. {citation}: {document.content[:240].strip()}..."
                )

            state["legal_opinion"] = "\n".join(report_lines)
            state["citations"] = citations

        return state

class GraderNode(Node):
    def __init__(self):
        super().__init__("GraderNode")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def execute(self, state: LegalState) -> LegalState:
        opinion = state.get("legal_opinion", "")
        documents = state.get("relevant_precedents", [])
        if not opinion:
            state["grade"] = "F"
            return state

        # Format documents for the prompt
        context_parts = []
        for index, document in enumerate(documents, start=1):
            citation = document.article_ref or f"{document.source}-page-{document.page}"
            context_parts.append(f"Document {index} ({citation}): {document.content}")

        context = "\n\n".join(context_parts)

        # Grading using Groq LLM
        prompt = f"""You are a strict Legal Auditor. Your only task is to determine if the "Generated Answer" is fully supported by the "Provided Legal Context."

### RULES
1. If every fact, date, and rule in the Answer is found in the Context, respond with: [PASS]
2. If the Answer contains information NOT found in the Context (hallucination), respond with: [FAIL]
3. If the Answer contradicts the Context, respond with: [FAIL]
4. Do not explain your reasoning unless you respond with [FAIL].

### PROVIDED LEGAL CONTEXT
{context}

### GENERATED ANSWER
{opinion}
"""

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500  # Increased to allow for explanation
            )

            result = response.choices[0].message.content.strip()

            if "[PASS]" in result:
                state["grade"] = "PASS"
                state["grader_feedback"] = None
            else:
                state["grade"] = "FAIL"
                state["grader_feedback"] = result  # Store the full response including reason
        except Exception as e:
            print(f"LLM grading failed: {e}, setting grade to UNKNOWN")
            state["grade"] = "UNKNOWN"
            state["grader_feedback"] = None

        return state

class WebSearchNode(Node):
    def __init__(self):
        super().__init__("WebSearchNode")

    def execute(self, state: LegalState) -> LegalState:
        plan = state.get("search_plan")
        if plan is None:
            raise ValueError("search_plan is required to determine web search behavior")

        if not plan.needs_web_search:
            return state

        state["legal_opinion"] = (
            "No matching legal documents were found in the local vector store. "
            "The system would now route to a public web search to find up-to-date legal updates."
        )
        state["citations"] = []
        return state
