"""Pydantic schemas shared across all agents."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Structured output of the Planner agent."""
    intent: str = Field(description="factual, comparative, explanatory, or analytical")
    entities: List[str] = Field(description="Key entities (people, products, dates)")
    complexity: str = Field(description="simple, moderate, or complex")
    sub_queries: Optional[List[str]] = Field(default=None, description="Sub-queries if complex")


class RetrievedChunk(BaseModel):
    """A retrieved document chunk with metadata."""
    doc_id: str
    text: str
    score: float
    metadata: Dict = Field(default_factory=dict)


class QualityGateResult(BaseModel):
    """Output of the Quality Gate agent."""
    passed: bool
    relevance_score: float
    coverage_score: float
    feedback: Optional[str] = None


class SynthesizedAnswer(BaseModel):
    """Final answer with citations."""
    response: str
    sources: List[str] = Field(description="Source document IDs cited")
    confidence: float = Field(description="Confidence between 0 and 1")


class CritiqueResult(BaseModel):
    """Output of the Critique agent."""
    passes: bool
    faithfulness_score: float
    completeness_score: float
    feedback: Optional[str] = None