"""
PDF Parser Module

Handles extraction of text and metadata from academic PDF papers.
"""

import fitz  # PyMuPDF
from typing import Dict, List, Tuple
from pathlib import Path


class PDFParser:
    """Parser for extracting text and metadata from PDF papers."""

    def __init__(self):
        pass

    def extract_text(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text and metadata from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing:
                - text: Full extracted text
                - pages: List of text per page
                - metadata: PDF metadata (title, author, etc.)
                - num_pages: Number of pages
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)

        # Extract text from all pages
        pages = []
        full_text = []

        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            pages.append({
                'page_num': page_num,
                'text': page_text
            })
            full_text.append(page_text)

        # Extract metadata
        metadata = {
            'title': doc.metadata.get('title', pdf_path.stem),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'keywords': doc.metadata.get('keywords', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
        }

        doc.close()

        return {
            'text': '\n\n'.join(full_text),
            'pages': pages,
            'metadata': metadata,
            'num_pages': len(pages),
            'file_path': str(pdf_path)
        }

    def extract_title_from_text(self, text: str) -> str:
        """
        Attempt to extract paper title from the first few lines.

        Args:
            text: Full text of the paper

        Returns:
            Extracted title (best guess)
        """
        lines = text.split('\n')
        # Usually title is in the first few non-empty lines
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:  # Reasonable title length
                return line
        return "Unknown Title"
