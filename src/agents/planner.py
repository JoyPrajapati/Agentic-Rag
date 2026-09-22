"""Planner agent - analyzes and decomposes queries."""

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.output import PromptedOutput

from src.agents.schemas import QueryAnalysis


class PlannerAgent:
    """Analyzes the user query to determine intent, entities, and complexity."""

    def __init__(self, model: str = "openrouter:deepseek/deepseek-r1"):
        self.agent = Agent(
            model,
            output_type=PromptedOutput(QueryAnalysis),
            retries={'output': 3},
            model_settings=ModelSettings(max_tokens=2000, temperature=0.1),
            system_prompt="""
            You are a Query Planner for an enterprise document assistant.

            Analyze the user's question and return:
            1. intent: one of "factual", "comparative", "explanatory", "analytical"
            2. entities: list of key people, products, dates, or departments mentioned
            3. complexity: "simple" (single lookup), "moderate" (2 hops), "complex" (multi-hop)
            4. sub_queries: if complexity is moderate or complex, break the question into 2-4 sub-questions.
               Otherwise return null.

            Be concise and accurate.
            """,
        )

    async def analyze(self, query: str) -> QueryAnalysis:
        result = await self.agent.run(query)
        return result.output