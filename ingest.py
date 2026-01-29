"""
loom/ingest.py

Module for ingesting documents from a specified directory. 
Supports Markdown, TXT, and PDF file types.
"""

import os
from pathlib import Path
from datetime import datetime
from .models import Document
import fitz  # PyMuPDF

class Ingestor:
    """
    Ingestor class to read and process documents.
    """

    @staticmethod
    def ingest_from_directory(directory: Path):
        """Ingest all supported files in the given directory."""
        docs = []
        for file_path in directory.glob("*"):
            if file_path.is_file():
                if file_path.suffix in ['.md', '.txt', '.pdf']:
                    if file_path.suffix == '.pdf':
                        raw_text = Ingestor.extract_pdf(file_path)
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            raw_text = f.read()
                    
                    content_hash = Document.compute_hash(raw_text)
                    created_at = datetime.fromtimestamp(file_path.stat().st_mtime)
                    doc_type = file_path.suffix[1:]  # remove the leading dot
                    docs.append(Document(path=file_path, raw_text=raw_text,
                                         content_hash=content_hash, created_at=created_at,
                                         file_type=doc_type))

        return docs
    
    @staticmethod
    def extract_pdf(file_path: Path) -> str:
        """Extract raw text from a PDF document."""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text