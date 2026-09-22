"""Long-term memory stored in a separate ChromaDB collection."""

from datetime import datetime
from typing import List

import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class LongTermMemory:
    """Persistent cross-session memory, using ChromaDB + vector similarity."""

    def __init__(
        self,
        chroma_client: chromadb.PersistentClient,
        collection_name: str = "conversation_memory",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
    ):
        self.client = chroma_client
        self.collection_name = collection_name
        self.embedder = HuggingFaceEmbedding(model_name=embedding_model)

        # Get or create the memory collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
        )

    def store_memory(self, user_id: str, key: str, value: str) -> None:
        """Store a memory entry."""
        embedding = self.embedder.get_query_embedding(value)

        self.collection.upsert(
            ids=[f"{user_id}_{key}"],
            embeddings=[embedding],
            documents=[value],
            metadatas=[{
                "user_id": user_id,
                "key": key,
                "timestamp": datetime.now().isoformat(),
            }],
        )

    def retrieve_memory(
        self, user_id: str, query: str, top_k: int = 3
    ) -> List[str]:
        """Retrieve relevant memories for a user."""
        query_embedding = self.embedder.get_query_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": user_id},
        )

        documents = results.get("documents", [[]])[0]
        return documents