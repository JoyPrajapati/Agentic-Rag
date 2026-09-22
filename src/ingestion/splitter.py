"""Document chunking with sentence-aware boundaries."""

from typing import List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode


class DocumentSplitter:
    """Split documents into semantic chunks."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? "]

        self.parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
            include_metadata=True,
        )

    def split_documents(self, documents: List[Document]) -> List[BaseNode]:
        """Split a list of documents into nodes (chunks)."""
        nodes = self.parser.get_nodes_from_documents(documents)
        print(f"Split {len(documents)} document(s) into {len(nodes)} node(s)")
        return nodes

    def split_single_document(self, document: Document) -> List[BaseNode]:
        """Split a single document into nodes."""
        return self.parser.get_nodes_from_documents([document])