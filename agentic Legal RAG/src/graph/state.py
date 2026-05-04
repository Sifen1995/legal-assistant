from typing import Annotated, List, Optional, TypedDict
from pydantic import BaseModel, Field
import operator

# --- 1. Pydantic Models for Evidence Tracking ---

class LegalDocument(BaseModel):
    """Represents a piece of legal text retrieved from the database."""
    content: str = Field(description="The actual text of the legal article/clause.")
    source: str = Field(description="The name of the PDF or Document.")
    page: Optional[int] = Field(None, description="Page number if available.")
    article_ref: Optional[str] = Field(None, description="The specific Article or Section number.")
    score: float = 0.0

class QueryPlan(BaseModel):
    """The structured output from the Query Analysis node."""
    refined_query: str
    metadata_filters: dict = Field(default_factory=dict)
    needs_web_search: bool = False

# --- 2. The Reducer Function ---
# This allows multiple retrieval steps to 'add' documents to the list 
# without overwriting previous findings.
def add_documents(existing: List[LegalDocument], new: List[LegalDocument]) -> List[LegalDocument]:
    return existing + new

# --- 3. The LangGraph State Definition ---

class LegalState(TypedDict):
    # User Inputs
    initial_query: str
    
    # Processed Intent (The 'Search Plan')
    search_plan: Optional[QueryPlan]
    
    # Evidence Collection 
    # (Annotated with our reducer to allow multi-step gathering)
    retrieved_docs: Annotated[List[LegalDocument], add_documents]
    
    # Refined Evidence (After the Re-ranker node filters the noise)
    relevant_precedents: List[LegalDocument]
    
    # Final Outputs
    legal_opinion: str
    citations: List[str]
     # For potential future use in grading the quality of the opinion
    grader_feedback: Optional[str] = None