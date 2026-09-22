"""Evaluate the RAG system against a golden dataset using RAGAS."""

import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

# --- Load .env BEFORE importing pydantic_ai ---
from dotenv import load_dotenv
load_dotenv()

# --- Project root on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Golden dataset path ---
GOLDEN_DATASET_PATH = PROJECT_ROOT / "tests" / "golden_datasets.json"

# --- Silence RAGAS deprecation noise ---
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

# --- RAGAS imports ---
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from openai import AsyncOpenAI
from datasets import Dataset

from src.agents.critique import CritiqueAgent
from src.agents.orchestrator import OrchestratorAgent, OrchestratorDeps
from src.agents.planner import PlannerAgent
from src.agents.quality_gate import QualityGateAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.synthesizer import SynthesizerAgent
from src.ingestion.indexer import DocumentIndexer
from src.memory.short_term import ShortTermMemory
from src.memory.working_context import WorkingContextManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


# ============================================================
# RAGAS CONFIG — DeepSeek judge + cached local embeddings
# ============================================================

# DeepSeek chat model is best for RAGAS judging (fast, structured JSON output).
# For deeper reasoning at the cost of speed, swap to:
#   "deepseek/deepseek-r1:free"
FREE_JUDGE_MODEL = "deepseek/deepseek-chat-v3-0324:free"

# Reuse the embedding model already cached on your disk.
# Zero new downloads, zero new disk usage.
EXISTING_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def get_ragas_llm():
    """DeepSeek judge via OpenRouter, using your existing OpenRouter key."""
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    return llm_factory(
        FREE_JUDGE_MODEL,
        client=client,
)


def get_ragas_embeddings():
    """Reuse cached BAAI/bge-large-en-v1.5 embeddings via RAGAS native HuggingFace."""
    return embedding_factory("huggingface", model=EXISTING_EMBEDDING_MODEL)


# ============================================================
# ORCHESTRATOR (ChromaDB version)
# ============================================================

async def build_orchestrator():
    """Rebuild the orchestrator with ChromaDB (no Qdrant)."""
    indexer = DocumentIndexer()
    chroma_client = indexer.get_client()      # ← ChromaDB client now
    embed_model = indexer.embed_model

    hybrid = HybridRetriever(
        chroma_client=chroma_client,          # ← parameter renamed
        collection_name="enterprise_documents",
        embed_model=embed_model,
        top_k=10,
    )
    reranker = CrossEncoderReranker()

    deps = OrchestratorDeps(
        planner=PlannerAgent(),
        retriever=RetrieverAgent(hybrid, reranker),
        synthesizer=SynthesizerAgent(),
        quality_gate=QualityGateAgent(),
        critique=CritiqueAgent(),
        memory=ShortTermMemory(),
        context_manager=WorkingContextManager(max_prompt_tokens=6000),
    )
    return OrchestratorAgent(deps)


# ============================================================
# MAIN
# ============================================================

async def main():
    if not GOLDEN_DATASET_PATH.exists():
        print(f"ERROR: Golden dataset not found at {GOLDEN_DATASET_PATH}")
        print("Create tests/golden_datasets.json with 20-30 Q&A pairs.")
        return

    golden = json.loads(GOLDEN_DATASET_PATH.read_text())
    print(f"Loaded {len(golden)} golden Q&A pairs.\n")

    orchestrator = await build_orchestrator()

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(golden, 1):
        q = item["question"]
        gt = item["ground_truth"]

        print(f"[{i}/{len(golden)}] {q[:70]}...")

        # Retrieval-only (for retrieval metrics)
        chunks = await orchestrator.deps.retriever.retrieve(q, top_k=5)
        ctx = [c.text for c in chunks]

        # Full pipeline (for generation metrics)
        result = await orchestrator.process_query(q, user_id="eval")

        questions.append(q)
        answers.append(result.response)
        contexts.append(ctx)
        ground_truths.append(gt)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nBuilding RAGAS judge (DeepSeek) and reusing cached embeddings...")
    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    print(f"Running RAGAS with '{FREE_JUDGE_MODEL}'...")
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,   # Let NaN happen instead of crashing
    )

    # --- Extract scores correctly (RAGAS 0.4+ returns EvaluationResult) ---
    scores_df = scores.to_pandas()

    aggregate = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        try:
            val = scores[metric]
            aggregate[metric] = float(val) if val is not None else None
        except Exception:
            aggregate[metric] = None

    print("\n" + "=" * 60)
    print("RAGAS SCORES")
    print("=" * 60)
    for metric, value in aggregate.items():
        if value is None or value != value:   # NaN check
            print(f"  {metric:20s}: ⚠️  NaN (judge failed on some samples)")
        else:
            print(f"  {metric:20s}: {value:.4f}")
    print("=" * 60)

    # Save aggregate results
    out_path = PROJECT_ROOT / "tests" / "ragas_results.json"
    out_path.write_text(json.dumps(aggregate, indent=2, default=str))
    print(f"\nAggregate scores saved to {out_path}")

    # Save per-sample CSV for debugging
    detail_path = PROJECT_ROOT / "tests" / "ragas_per_sample.csv"
    scores_df.to_csv(detail_path, index=False)
    print(f"Per-sample scores saved to {detail_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        import gc
        gc.collect()   # suppresses the Qdrant __del__ noise