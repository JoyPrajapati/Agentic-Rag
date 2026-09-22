"""Index documents into ChromaDB (local persistent storage)."""

from typing import List
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import BaseNode


# Project root = 3 levels up from this file
# (indexer.py -> ingestion -> src -> project_root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DocumentIndexer:
    """Index documents into a local ChromaDB persistent storage folder."""

    def __init__(
        self,
        collection_name: str = "enterprise_documents",
        storage_path: str = str(PROJECT_ROOT / "chroma_db"),
        embedding_model: str = "BAAI/bge-large-en-v1.5",
    ):
        self.collection_name = collection_name
        self.storage_path = storage_path

        # Load embedding model (reuses the cached ~1.3GB model)
        print(f"Loading embedding model: {embedding_model}...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            trust_remote_code=True,
        )
        Settings.embed_model = self.embed_model

        # 🔥 ChromaDB persistent client (anti-docker, no server)
        print(f"Connecting to ChromaDB at: {storage_path}")
        self.chroma_client = chromadb.PersistentClient(path=storage_path)

        # Create or get collection
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
        )

        # Wrap in LlamaIndex vector store
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
        )

    def index_nodes(self, nodes: List[BaseNode]) -> None:
        """Index nodes into ChromaDB."""
        VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            store_nodes_override=True,
        )
        print(f"Indexed {len(nodes)} nodes into ChromaDB at '{self.storage_path}'")

    def get_index(self) -> VectorStoreIndex:
        """Get the existing index."""
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

    def get_client(self) -> chromadb.PersistentClient:
        """Return the ChromaDB client for use in retrieval."""
        return self.chroma_client

    def get_collection(self):
        """Return the ChromaDB collection for direct access."""
        return self.chroma_collection