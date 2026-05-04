from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field

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

# --- 2. The State Definition ---

class LegalState(TypedDict):
    # User Inputs
    initial_query: str
    
    # Processed Intent (The 'Search Plan')
    search_plan: Optional[QueryPlan]
    
    # Evidence Collection
    retrieved_docs: List[LegalDocument]
    
    # Refined Evidence (After the Re-ranker node filters the noise)
    relevant_precedents: List[LegalDocument]
    
    # Final Outputs
    legal_opinion: str
    citations: List[str]
    grade: Optional[str]
     # For potential future use in grading the quality of the opinion
    grader_feedback: Optional[str] = None