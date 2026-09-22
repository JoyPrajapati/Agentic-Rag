"""Orchestrator - coordinates all agents in a deterministic flow."""

from src.agents.critique import CritiqueAgent
from src.agents.planner import PlannerAgent
from src.agents.quality_gate import QualityGateAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.schemas import SynthesizedAnswer
from src.agents.synthesizer import SynthesizerAgent
from src.memory.short_term import ShortTermMemory
from src.memory.working_context import WorkingContextManager


class OrchestratorDeps:
    """Dependency container for the orchestrator."""

    def __init__(
        self,
        planner: PlannerAgent,
        retriever: RetrieverAgent,
        synthesizer: SynthesizerAgent,
        quality_gate: QualityGateAgent,
        critique: CritiqueAgent,
        memory: ShortTermMemory,
        context_manager: WorkingContextManager,
    ):
        self.planner = planner
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.quality_gate = quality_gate
        self.critique = critique
        self.memory = memory
        self.context_manager = context_manager


class OrchestratorAgent:
    """Main orchestrator. Runs the full pipeline for each query."""

    def __init__(self, deps: OrchestratorDeps, max_critique_loops: int = 2):
        self.deps = deps
        self.max_critique_loops = max_critique_loops

    async def process_query(
        self, query: str, user_id: str = "default"
    ) -> SynthesizedAnswer:
        """Run the full agentic pipeline."""

        # 1. Fetch conversation history
        history = self.deps.memory.get_history(user_id)

        # 2. Plan (analyze query)
        print("\n[Orchestrator] Step 1: Planning...")
        plan = await self.deps.planner.analyze(query)
        print(f"  Intent: {plan.intent} | Complexity: {plan.complexity}")

        # 3. Retrieve
        print("[Orchestrator] Step 2: Retrieving documents...")
        chunks = await self.deps.retriever.retrieve(query, top_k=10)

        if not chunks:
            return SynthesizedAnswer(
                response="I couldn't find any relevant documents to answer your question.",
                sources=[],
                confidence=0.0,
            )

        # 4. Quality Gate
        print("[Orchestrator] Step 3: Quality gate...")
        quality = await self.deps.quality_gate.evaluate(query, chunks)
        print(f"  Passed: {quality.passed} | Relevance: {quality.relevance_score:.2f}")
        if not quality.passed:
            print(f"  Feedback: {quality.feedback}")

        # 5. Trim context
        print("[Orchestrator] Step 4: Trimming context...")
        trimmed = self.deps.context_manager.prepare_context(
            query=query,
            retrieved_chunks=chunks,
            conversation_history=history,
        )

        # 6. Synthesize
        print("[Orchestrator] Step 5: Synthesizing answer...")
        answer = await self.deps.synthesizer.synthesize_with_context(
            query=query, context=trimmed
        )

        # 7. Critique loop
        print("[Orchestrator] Step 6: Critiquing answer...")
        for attempt in range(self.max_critique_loops):
            critique = await self.deps.critique.critique(
                query=query, answer=answer.response, chunks=chunks
            )
            if critique.passes:
                print("  Critique passed.")
                break
            print(f"  Critique failed (attempt {attempt + 1}): {critique.feedback}")
            answer = await self.deps.synthesizer.synthesize_with_context(
                query=query,
                context=trimmed + f"\n\nReviewer feedback to address: {critique.feedback}",
            )

        # 8. Store memory
        self.deps.memory.add_turn(user_id, query, answer.response)

        print("[Orchestrator] ✅ Done.\n")
        return answer