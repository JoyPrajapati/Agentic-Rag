"""Ingestion script - run this once to index your documents."""

import asyncio
from src.ingestion.loader import DocumentLoader
from src.ingestion.splitter import DocumentSplitter
from src.ingestion.indexer import DocumentIndexer


async def main():
    print("=" * 60)
    print("STEP 1: Loading documents from data/raw/")
    print("=" * 60)
    loader = DocumentLoader()
    docs = loader.load_from_directory("data/raw")

    if not docs:
        print("ERROR: No documents found in data/raw/. Please add PDFs/TXTs.")
        return

    print("\n" + "=" * 60)
    print(f"STEP 2: Splitting {len(docs)} documents into chunks")
    print("=" * 60)
    splitter = DocumentSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.split_documents(docs)

    print("\n" + "=" * 60)
    print(f"STEP 3: Indexing {len(nodes)} chunks into local Qdrant storage")
    print("=" * 60)
    indexer = DocumentIndexer()  # Local storage mode
    indexer.index_nodes(nodes)

    print("\n" + "=" * 60)
    print("✅ Ingestion complete! You can now run the API.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())