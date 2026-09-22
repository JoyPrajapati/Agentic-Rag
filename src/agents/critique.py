"""Critique agent - reviews answers for faithfulness and completeness."""

from typing import List

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings   
from pydantic_ai.output import PromptedOutput 

from src.agents.schemas import CritiqueResult, RetrievedChunk


class CritiqueAgent:
    """Reviews a synthesized answer before returning to the user."""

    def __init__(self, model: str = "openrouter:deepseek/deepseek-r1"):
        self.agent = Agent(
            model,
            output_type=PromptedOutput(CritiqueResult),
            retries={'output': 3},
            model_settings=ModelSettings(max_tokens=2000, temperature=0.1),
            system_prompt="""
            You are a Critique Agent. Review the synthesized answer against the context.

            Evaluate:
            1. faithfulness_score (0.0 - 1.0): Is every claim supported by the context?
            2. completeness_score (0.0 - 1.0): Does the answer address all parts of the question?
            3. passes: True if faithfulness_score >= 0.8 AND completeness_score >= 0.7.
            4. feedback: If passes is False, explain what needs fixing in 1-2 sentences.
                       Otherwise null.

            Be strict about hallucination.
            """,
        )

    async def critique(
        self, query: str, answer: str, chunks: List[RetrievedChunk]
    ) -> CritiqueResult:
        context = "\n\n".join([c.text for c in chunks[:5]])
        prompt = (
            f"Query: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Generated Answer:\n{answer}"
        )
        result = await self.agent.run(prompt)
        return result.output