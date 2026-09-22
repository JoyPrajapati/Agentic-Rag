"""Dump the raw extracted text from the first chunk of each PDF."""

from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_dir="data/raw",
    filename_as_id=True,
)
docs = reader.load_data()

for doc in docs[:2]:
    print("=" * 70)
    print(f"FILE: {doc.metadata.get('file_name', 'unknown')}")
    print("=" * 70)
    print(doc.text[:500])
    print("...")
    print()