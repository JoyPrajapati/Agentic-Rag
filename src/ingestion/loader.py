"""Document loader with robust PDF parsing via PyMuPDF."""

from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from llama_index.core.schema import Document


class DocumentLoader:
    """Load PDFs via PyMuPDF and text files directly."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".csv"}

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_pdf(self, file_path: Path) -> Document:
        """Extract text from a PDF using PyMuPDF."""
        doc = fitz.open(str(file_path))
        pages_text = []

        for page_num, page in enumerate(doc, start=1):
            # "text" mode preserves reading order better than "blocks"
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)

        doc.close()
        combined = "\n\n".join(pages_text)

        return Document(
            text=combined,
            metadata={
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": "pdf",
                "num_pages": len(pages_text),
            },
            id_=str(file_path),
        )

    def _load_text_file(self, file_path: Path) -> Document:
        """Load a plain text / markdown / CSV file."""
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return Document(
            text=text,
            metadata={
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lstrip("."),
            },
            id_=str(file_path),
        )

    def load_from_directory(
        self,
        directory: Optional[str] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """Load all supported documents from a directory."""
        target_dir = Path(directory) if directory else self.data_dir

        if not target_dir.exists():
            raise FileNotFoundError(f"Directory not found: {target_dir}")

        pattern = "**/*" if recursive else "*"
        documents: List[Document] = []

        for file_path in sorted(target_dir.glob(pattern)):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            try:
                ext = file_path.suffix.lower()
                if ext == ".pdf":
                    doc = self._load_pdf(file_path)
                elif ext in {".txt", ".md", ".csv"}:
                    doc = self._load_text_file(file_path)
                else:
                    # Fallback for .docx etc.
                    from llama_index.core import SimpleDirectoryReader
                    sub_docs = SimpleDirectoryReader(
                        input_files=[str(file_path)]
                    ).load_data()
                    for sd in sub_docs:
                        documents.append(sd)
                    print(f"  ✓ Loaded {file_path.name} via fallback reader")
                    continue

                if doc.text.strip():
                    documents.append(doc)
                    print(f"  ✓ Loaded {file_path.name} ({len(doc.text):,} chars)")
                else:
                    print(f"  ⚠ {file_path.name} produced empty text")

            except Exception as e:
                print(f"  ✗ Failed to load {file_path.name}: {e}")

        print(f"Loaded {len(documents)} document(s) from {target_dir}")
        return documents

    def load_single_file(self, file_path: str) -> List[Document]:
        """Load a single file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return [self._load_pdf(path)]
        elif ext in {".txt", ".md", ".csv"}:
            return [self._load_text_file(path)]
        else:
            from llama_index.core import SimpleDirectoryReader
            return SimpleDirectoryReader(input_files=[str(path)]).load_data()

    def load_from_text(self, text: str, metadata: Optional[dict] = None) -> Document:
        """Load from a raw text string."""
        return Document(
            text=text,
            metadata=metadata or {"source": "raw_text"},
        )