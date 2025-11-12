"""
Process all PDFs in the data/papers directory

This will add them all to the library database.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_parser import PDFParser
from src.library import PaperLibrary


def process_all_papers():
    """Process all PDF papers in data/papers directory."""

    papers_dir = Path("data/papers")

    if not papers_dir.exists():
        print("Error: data/papers directory not found!")
        return

    pdf_files = list(papers_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/papers/")
        return

    print("=" * 60)
    print(f"Found {len(pdf_files)} PDF files to process")
    print("=" * 60)
    print()

    parser = PDFParser()
    library = PaperLibrary()

    # Check which papers are already in the library
    existing_papers = library.list_papers()
    existing_paths = {p['file_path'] for p in existing_papers}

    processed = 0
    skipped = 0
    errors = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

        # Check if already processed
        if str(pdf_path) in existing_paths:
            print(f"      SKIPPED - Already in library")
            skipped += 1
            print()
            continue

        try:
            # Parse the PDF
            parsed_data = parser.extract_text(str(pdf_path))

            # Extract title (try metadata first, then text)
            title = parsed_data['metadata'].get('title')
            if not title or len(title) < 10:
                title = parser.extract_title_from_text(parsed_data['text'])

            authors = parsed_data['metadata'].get('author', '')

            # Add to library
            paper_id = library.add_paper(
                title=title,
                file_path=str(pdf_path),
                authors=authors,
                num_pages=parsed_data['num_pages']
            )

            print(f"      SUCCESS - Paper ID: {paper_id}")
            print(f"      Title: {title[:70]}...")
            print(f"      Pages: {parsed_data['num_pages']}")
            processed += 1

        except Exception as e:
            print(f"      ERROR: {str(e)}")
            errors += 1

        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Skipped (already in library): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total in library: {len(library.list_papers())}")
    print()

    if processed > 0:
        print("Next step: Index these papers (Phase 2 - coming soon!)")
        print()


if __name__ == "__main__":
    process_all_papers()
