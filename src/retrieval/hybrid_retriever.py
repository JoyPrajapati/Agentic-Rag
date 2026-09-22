"""Hybrid retrieval: dense (ChromaDB) + sparse (BM25) with RRF fusion."""

from typing import List, Tuple

import numpy as np
import chromadb
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """Combines dense semantic search with BM25 keyword search."""

    def __init__(
        self,
        chroma_client: chromadb.PersistentClient,
        collection_name: str,
        embed_model,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        k_rrf: int = 60,
    ):
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.k_rrf = k_rrf

        # Get the raw Chroma collection for direct access
        self.chroma_collection = chroma_client.get_or_create_collection(
            name=collection_name,
        )

        # Wrap in LlamaIndex vector store (for consistency)
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection,
        )

    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Semantic search using dense embeddings via ChromaDB."""
        query_embedding = self.embed_model.get_query_embedding(query)

        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "metadatas", "documents"],
        )

        # Chroma returns distances (lower = better). Convert to similarity scores.
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Convert distance to similarity: similarity = 1 / (1 + distance)
        hits = []
        for doc_id, distance in zip(ids, distances):
            similarity = 1.0 / (1.0 + float(distance))
            hits.append((doc_id, similarity))

        return hits

    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Keyword search using BM25 over all documents in the collection."""
        # Fetch all documents from Chroma
        all_data = self.chroma_collection.get(
            include=["documents", "metadatas"],
        )

        ids = all_data.get("ids", [])
        documents = all_data.get("documents", [])

        if not ids or not documents:
            print("BM25: No documents found in ChromaDB collection.")
            return []

        # Build BM25 index
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            (ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Combine results using Reciprocal Rank Fusion."""
        scores = {}

        for rank, (doc_id, _) in enumerate(dense_results):
            scores[doc_id] = scores.get(doc_id, 0) + self.dense_weight / (
                self.k_rrf + rank + 1
            )

        for rank, (doc_id, _) in enumerate(sparse_results):
            scores[doc_id] = scores.get(doc_id, 0) + self.sparse_weight / (
                self.k_rrf + rank + 1
            )

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str) -> List[NodeWithScore]:
        """Run hybrid retrieval and return nodes with scores."""
        # Dense search
        dense_results = self.dense_search(query, self.top_k)

        # Sparse search (BM25)
        sparse_results = self.sparse_search(query, self.top_k)

        # Fuse
        fused_results = self.reciprocal_rank_fusion(
            dense_results, sparse_results
        )
        top_results = fused_results[: self.top_k]

        # Fetch the actual text for each retrieved doc
        nodes_with_score = []
        if top_results:
            doc_ids = [doc_id for doc_id, _ in top_results]

            # Fetch documents by IDs from Chroma
            fetched = self.chroma_collection.get(
                ids=doc_ids,
                include=["documents", "metadatas"],
            )

            id_to_doc = {}
            for i, doc_id in enumerate(fetched.get("ids", [])):
                docs = fetched.get("documents", [])
                metas = fetched.get("metadatas", [])
                text = docs[i] if i < len(docs) else ""
                metadata = metas[i] if i < len(metas) else {}
                id_to_doc[doc_id] = (text, metadata)

            for doc_id, score in top_results:
                if doc_id in id_to_doc:
                    text, metadata = id_to_doc[doc_id]
                    node = TextNode(
                        text=text,
                        metadata=metadata or {},
                        id_=doc_id,
                    )
                    nodes_with_score.append(
                        NodeWithScore(node=node, score=score)
                    )

        return nodes_with_score