"""
List all papers in the library

Usage:
    python list_papers.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.library import PaperLibrary


def list_papers():
    """List all papers in the library."""

    library = PaperLibrary()
    papers = library.list_papers()

    if not papers:
        print("=" * 60)
        print("No papers in library yet!")
        print("=" * 60)
        print()
        print("To add papers:")
        print("  1. Place PDF files in: data/papers/")
        print("  2. Or run: python upload_paper.py path/to/paper.pdf")
        print()
        return

    print("=" * 60)
    print(f"Papers in Library ({len(papers)} total)")
    print("=" * 60)
    print()

    for paper in papers:
        print(f"ID: {paper['id']}")
        print(f"Title: {paper['title']}")
        print(f"Authors: {paper['authors'] or 'Unknown'}")
        print(f"Pages: {paper['num_pages']}")
        print(f"File: {paper['file_path']}")
        print(f"Added: {paper['indexed_date']}")

        # Check if file exists
        file_path = Path(paper['file_path'])
        if not file_path.exists():
            print("  WARNING: File not found!")

        print("-" * 60)
        print()


if __name__ == "__main__":
    list_papers()
