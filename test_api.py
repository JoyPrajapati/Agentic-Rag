"""Quick API sanity check."""
import asyncio
from dotenv import load_dotenv
load_dotenv()

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic import BaseModel


class TestOutput(BaseModel):
    status: str
    answer: int


async def main():
    agent = Agent(
        "openrouter:openai/gpt-4o-mini",
        output_type=TestOutput,
        model_settings=ModelSettings(max_tokens=100, temperature=0.1),
        system_prompt="You are a test agent. Return status='OK' and answer=42.",
    )
    result = await agent.run("Run the test.")
    print(f"✅ SUCCESS: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())