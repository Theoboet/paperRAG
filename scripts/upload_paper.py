"""
Upload and process a PDF paper

Usage:
    python upload_paper.py path/to/paper.pdf
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_parser import PDFParser
from src.library import PaperLibrary


def upload_paper(pdf_path: str):
    """Upload and process a PDF paper."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return

    if pdf_path.suffix.lower() != '.pdf':
        print(f"Error: File must be a PDF: {pdf_path}")
        return

    print("=" * 60)
    print("Uploading paper to paperRAG")
    print("=" * 60)
    print()

    # Initialize components
    parser = PDFParser()
    library = PaperLibrary()

    # Parse the PDF
    print(f"[1/3] Parsing PDF: {pdf_path.name}")
    try:
        parsed_data = parser.extract_text(str(pdf_path))
        print(f"      Extracted {parsed_data['num_pages']} pages")
        print(f"      Title: {parsed_data['metadata']['title'][:60]}...")
        print()
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return

    # Copy to data/papers directory
    print("[2/3] Copying to data/papers/")
    data_papers_dir = Path("data/papers")
    data_papers_dir.mkdir(parents=True, exist_ok=True)

    destination = data_papers_dir / pdf_path.name

    # Copy file if not already there
    if not destination.exists():
        import shutil
        shutil.copy2(pdf_path, destination)
        print(f"      Copied to: {destination}")
    else:
        print(f"      Already exists: {destination}")
    print()

    # Add to library
    print("[3/3] Adding to library database")
    try:
        title = parsed_data['metadata'].get('title') or parser.extract_title_from_text(parsed_data['text'])
        authors = parsed_data['metadata'].get('author', '')

        paper_id = library.add_paper(
            title=title,
            file_path=str(destination),
            authors=authors,
            num_pages=parsed_data['num_pages']
        )

        print(f"      Paper ID: {paper_id}")
        print(f"      Title: {title}")
        print(f"      Authors: {authors or 'Unknown'}")
        print()
    except Exception as e:
        print(f"Error adding to library: {e}")
        return

    print("=" * 60)
    print("SUCCESS! Paper uploaded and ready for indexing")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. The paper is now in the library (ID: {paper_id})")
    print("  2. Next, you'll need to index it (Phase 2)")
    print("  3. Then you can ask questions about it!")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_paper.py path/to/paper.pdf")
        print()
        print("Example:")
        print("  python upload_paper.py ~/Downloads/attention_is_all_you_need.pdf")
        print()
        print("Where to find papers:")
        print("  - arXiv.org (download PDFs)")
        print("  - Your local collection")
        print()
        return

    pdf_path = sys.argv[1]
    upload_paper(pdf_path)


if __name__ == "__main__":
    main()
