"""
Main entry point for paperRAG

This will be developed as we progress through the implementation phases.
"""

from src.pdf_parser import PDFParser
from src.library import PaperLibrary
from src.indexer import TextChunker, VectorIndexer
from src.retriever import Retriever
from src.generator import AnswerGenerator


def main():
    """Main application entry point."""
    print("=" * 60)
    print("paperRAG - Offline Assistant for Computer Vision Papers")
    print("=" * 60)
    print()
    print("Status: Phase 1.1 - Foundation Complete")
    print()
    print("Modules initialized:")
    print("  [OK] PDF Parser")
    print("  [OK] Paper Library (SQLite)")
    print("  [OK] Text Chunker")
    print("  [OK] Vector Indexer (ChromaDB)")
    print("  [OK] Retriever")
    print("  [OK] Answer Generator (placeholder)")
    print()
    print("Next steps:")
    print("  - Test PDF parsing with real papers (Step 1.2)")
    print("  - Test library management (Step 1.3)")
    print("  - Implement RAG core functionality (Phase 2)")
    print()


if __name__ == "__main__":
    main()
