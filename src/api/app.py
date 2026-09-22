"""FastAPI application for the Enterprise Knowledge Assistant (ChromaDB Local Mode)."""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

load_dotenv()

# --------------------------------------------------------------------
# Pydantic models for API
# --------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default"


class Source(BaseModel):
    doc_id: str


class QueryResponse(BaseModel):
    response: str
    sources: List[Source]
    confidence: float


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    nodes_created: int


# --------------------------------------------------------------------
# Global orchestrator
# --------------------------------------------------------------------
orchestrator: Optional[OrchestratorAgent] = None


# --------------------------------------------------------------------
# Lifespan: initialize all components at startup
# --------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator

    try:
        logfire.configure()
        logfire.instrument_pydantic_ai()
    except Exception as e:
        print(f"[Startup] Logfire not configured: {e}")

    print("\n" + "=" * 60)
    print("🚀 Initializing Enterprise Knowledge Assistant (ChromaDB)")
    print("=" * 60)

    # 1. Connect to ChromaDB via the indexer
    indexer = DocumentIndexer()
    chroma_client = indexer.get_client()
    embed_model = indexer.embed_model

    # 2. Retrieval (ChromaDB version)
    hybrid_retriever = HybridRetriever(
        chroma_client=chroma_client,
        collection_name="enterprise_documents",
        embed_model=embed_model,
        top_k=10,
    )
    reranker = CrossEncoderReranker()

    # 3. Agents
    planner = PlannerAgent()
    retriever_agent = RetrieverAgent(hybrid_retriever, reranker)
    synthesizer = SynthesizerAgent()
    quality_gate = QualityGateAgent()
    critique = CritiqueAgent()

    # 4. Memory
    memory = ShortTermMemory(max_turns=5)
    context_manager = WorkingContextManager(max_prompt_tokens=6000)

    # 5. Orchestrator
    deps = OrchestratorDeps(
        planner=planner,
        retriever=retriever_agent,
        synthesizer=synthesizer,
        quality_gate=quality_gate,
        critique=critique,
        memory=memory,
        context_manager=context_manager,
    )
    orchestrator = OrchestratorAgent(deps)

    print("✅ Ready to serve requests.\n")
    yield

    print("\n🛑 Shutting down...")


# --------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------
app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="Agentic Document RAG (ChromaDB Local Storage)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "healthy", "service": "Enterprise Knowledge Assistant"}


@app.get("/v1/health")
async def health():
    return {"status": "healthy"}


@app.post("/v1/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await orchestrator.process_query(
            query=request.query, user_id=request.user_id
        )
        return QueryResponse(
            response=result.response,
            sources=[Source(doc_id=s) for s in result.sources],
            confidence=result.confidence,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest_documents():
    from src.ingestion.loader import DocumentLoader
    from src.ingestion.splitter import DocumentSplitter
    from src.ingestion.indexer import DocumentIndexer

    try:
        loader = DocumentLoader()
        splitter = DocumentSplitter()
        indexer = DocumentIndexer()

        docs = loader.load_from_directory()
        nodes = splitter.split_documents(docs)
        indexer.index_nodes(nodes)

        return IngestResponse(
            status="success",
            documents_processed=len(docs),
            nodes_created=len(nodes),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )