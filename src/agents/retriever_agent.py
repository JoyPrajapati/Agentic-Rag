"""Retriever agent - wraps HybridRetriever and CrossEncoderReranker."""

from typing import List

from src.agents.schemas import RetrievedChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


class RetrieverAgent:
    """Retrieves and reranks relevant document chunks."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    async def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        """Retrieve candidates then rerank to top_k."""
        # Step 1: Hybrid retrieval (returns NodeWithScore)
        results = self.hybrid_retriever.retrieve(query)

        if not results:
            print("[RetrieverAgent] No results found.")
            return []

        # Step 2: Prepare for reranking
        doc_texts = [(r.node.id_, r.node.text) for r in results]

        # Step 3: Rerank
        reranked = self.reranker.rerank(query, doc_texts, top_k=top_k)

        # Step 4: Convert to RetrievedChunk
        chunks = [
            RetrievedChunk(doc_id=doc_id, text=text, score=score, metadata={})
            for doc_id, text, score in reranked
        ]
        print(f"[RetrieverAgent] Retrieved {len(results)} → reranked to {len(chunks)}")
        return chunks