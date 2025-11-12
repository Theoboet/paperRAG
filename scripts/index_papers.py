"""
Index all papers in the library

This creates embeddings for semantic search.
Run this after uploading papers to make them searchable.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.library import PaperLibrary
from src.pdf_parser import PDFParser
from src.indexer import TextChunker, VectorIndexer


def index_all_papers():
    """Index all papers in the library."""

    print("=" * 60)
    print("Indexing Papers - Creating Embeddings")
    print("=" * 60)
    print()

    # Initialize components
    library = PaperLibrary()
    parser = PDFParser()
    chunker = TextChunker(chunk_size=500, overlap=75)

    print("Initializing vector indexer (this may take a moment)...")
    indexer = VectorIndexer(
        collection_name="papers",
        persist_directory="data/vector_db"
    )
    print()

    # Get all papers
    papers = library.list_papers()

    if not papers:
        print("No papers found in library!")
        print("Run: python process_all_papers.py first")
        return

    print(f"Found {len(papers)} papers in library")
    print()

    total_chunks = 0
    successful = 0
    errors = 0

    for i, paper in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}] Indexing: {paper['title'][:60]}...")

        try:
            # Check if file exists
            file_path = Path(paper['file_path'])
            if not file_path.exists():
                print(f"      ERROR: File not found: {file_path}")
                errors += 1
                continue

            # Parse PDF to get text and pages
            parsed_data = parser.extract_text(str(file_path))

            # Create chunks with page metadata
            all_chunks = []
            for page in parsed_data['pages']:
                page_text = page['text']
                if not page_text.strip():
                    continue

                chunks = chunker.chunk_text(
                    text=page_text,
                    paper_id=paper['id'],
                    metadata={
                        'page_num': page['page_num'],
                        'title': paper['title'],
                        'authors': paper['authors']
                    }
                )
                all_chunks.extend(chunks)

            if not all_chunks:
                print(f"      WARNING: No chunks created (empty paper?)")
                continue

            # Index chunks
            indexer.index_chunks(all_chunks)

            print(f"      SUCCESS: {len(all_chunks)} chunks indexed")
            total_chunks += len(all_chunks)
            successful += 1

        except Exception as e:
            print(f"      ERROR: {str(e)}")
            errors += 1

        print()

    print("=" * 60)
    print("Indexing Complete!")
    print("=" * 60)
    print(f"Papers indexed: {successful}/{len(papers)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Errors: {errors}")
    print()

    if successful > 0:
        print("Your papers are now searchable!")
        print("Next: Run 'python search_papers.py \"your question\"'")
    print()


if __name__ == "__main__":
    index_all_papers()
