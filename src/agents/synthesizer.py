"""Synthesizer agent - generates grounded answers from context."""

from typing import List

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.output import PromptedOutput 

from src.agents.schemas import RetrievedChunk, SynthesizedAnswer


class SynthesizerAgent:
    """Generates the final answer from trimmed context."""

    def __init__(self, model: str = "openrouter:deepseek/deepseek-v4-flash"):
        self.agent = Agent(
            model,
            output_type=PromptedOutput(SynthesizedAnswer),
            retries={'output': 3},
            model_settings=ModelSettings(max_tokens=4000, temperature=0.1),
            system_prompt="""
            You are a Synthesis Agent for an enterprise document assistant.

            Your ONLY job is to answer the user's question based on the provided context.

            Rules:
            1. Use ONLY information present in the context.
            2. If the context does not contain the answer, respond with:
               "I don't have this information in the available documents."
            3. Cite sources using the [Source: doc_id] tags.
            4. Be concise but complete.
            5. Return confidence between 0.0 and 1.0 based on how well the context supports your answer.

            Never invent facts or use external knowledge.
            """,
        )

    async def synthesize(
        self, query: str, chunks: List[RetrievedChunk]
    ) -> SynthesizedAnswer:
        context = "\n\n".join(f"[Source: {c.doc_id}]\n{c.text}" for c in chunks)
        prompt = f"User Question: {query}\n\nRetrieved Context:\n{context}"
        result = await self.agent.run(prompt)
        return result.output

    async def synthesize_with_context(
        self, query: str, context: str
    ) -> SynthesizedAnswer:
        prompt = f"User Question: {query}\n\nContext (trimmed):\n{context}"
        result = await self.agent.run(prompt)
        return result.output