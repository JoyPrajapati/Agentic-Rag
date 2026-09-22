"""Quality Gate agent - scores retrieval relevance and coverage."""

from typing import List

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.output import PromptedOutput


from src.agents.schemas import QualityGateResult, RetrievedChunk


class QualityGateAgent:
    """Evaluates whether retrieved chunks are good enough to answer."""

    def __init__(self, model: str = "openrouter:deepseek/deepseek-v4-flash", threshold: float = 0.5):
        self.threshold = threshold
        self.agent = Agent(
            model,
            output_type=PromptedOutput(QualityGateResult),
            retries={'output': 3},
            model_settings=ModelSettings(max_tokens=2000, temperature=0.1),
            system_prompt="""
            You are a Quality Gate. Given a user query and retrieved document chunks, evaluate:

            1. relevance_score (0.0 - 1.0): How relevant are these chunks to the query?
            2. coverage_score (0.0 - 1.0): Do the chunks cover all aspects of the query?
            3. passed: True if relevance_score >= 0.5, False otherwise.
            4. feedback: If passed is False, briefly explain what's missing (1 sentence).
                       Otherwise return null.

            Be strict but fair. Only pass chunks that can actually answer the question.
            """,
        )

    async def evaluate(
        self, query: str, chunks: List[RetrievedChunk]
    ) -> QualityGateResult:
        if not chunks:
            return QualityGateResult(
                passed=False,
                relevance_score=0.0,
                coverage_score=0.0,
                feedback="No chunks were retrieved.",
            )

        context = "\n\n".join([c.text for c in chunks[:5]])
        prompt = f"Query: {query}\n\nRetrieved chunks:\n{context}"

        result = await self.agent.run(prompt)
        output = result.output
        output.passed = output.relevance_score >= self.threshold
        return output