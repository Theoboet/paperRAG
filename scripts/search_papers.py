"""
Search papers using semantic search

Usage:
    python search_papers.py "your question here"

Examples:
    python search_papers.py "what is attention mechanism"
    python search_papers.py "how does vision transformer work"
    python search_papers.py "what are the advantages of mamba"
"""

import sys
import io
from pathlib import Path

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexer import VectorIndexer
from src.retriever import Retriever
from src.library import PaperLibrary


def search_papers(query: str, top_k: int = 5):
    """Search for relevant passages in your paper library."""

    print("=" * 80)
    print("Semantic Search - paperRAG")
    print("=" * 80)
    print()
    print(f"Query: {query}")
    print()

    # Initialize components
    print("[1/3] Loading vector database and embedding model...")
    indexer = VectorIndexer(
        collection_name="papers",
        persist_directory="data/vector_db"
    )

    retriever = Retriever(indexer)
    library = PaperLibrary()
    print()

    # Perform search
    print(f"[2/3] Searching for top {top_k} relevant passages...")
    results = retriever.retrieve(query, top_k=top_k)
    print()

    if not results:
        print("No results found!")
        print("Make sure you've indexed your papers first:")
        print("  python scripts/index_papers.py")
        return

    # Display results
    print(f"[3/3] Found {len(results)} relevant passages:")
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    for i, result in enumerate(results, 1):
        # Get paper info
        paper_id = result['metadata'].get('paper_id', 'Unknown')
        page_num = result['metadata'].get('page_num', 'Unknown')
        title = result['metadata'].get('title', 'Unknown Title')

        # Get similarity score (distance - lower is better for cosine)
        distance = result.get('distance')
        if distance is not None:
            similarity = 1 - distance  # Convert distance to similarity
            similarity_pct = similarity * 100
        else:
            similarity_pct = None

        # Display result
        print(f"Result #{i}")
        print("-" * 80)
        print(f"Paper ID: {paper_id}")
        print(f"Title: {title}")
        print(f"Page: {page_num}")

        if similarity_pct is not None:
            print(f"Relevance: {similarity_pct:.1f}%")

        print()
        print("Text:")
        print(result['text'])
        print()
        print("=" * 80)
        print()

    # Show formatted context (what would be sent to LLM)
    print()
    print("FORMATTED CONTEXT FOR LLM")
    print("=" * 80)
    context = retriever.format_context(results)
    print(context)
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python search_papers.py \"your question\"")
        print()
        print("Examples:")
        print('  python search_papers.py "what is attention mechanism"')
        print('  python search_papers.py "how does vision transformer work"')
        print('  python search_papers.py "what are the advantages of mamba"')
        print('  python search_papers.py "what is histogram of oriented gradients"')
        print()
        return

    query = sys.argv[1]

    # Optional: specify top_k
    top_k = 5
    if len(sys.argv) > 2:
        try:
            top_k = int(sys.argv[2])
        except ValueError:
            print(f"Warning: Invalid top_k value '{sys.argv[2]}', using default of 5")

    search_papers(query, top_k)


if __name__ == "__main__":
    main()
