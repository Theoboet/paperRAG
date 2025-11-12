"""
Visualize paper embeddings in 2D space

This shows how similar papers cluster together based on their content.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexer import VectorIndexer
from src.library import PaperLibrary
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict


def visualize_embeddings():
    """Visualize embeddings using t-SNE dimensionality reduction."""

    print("=" * 60)
    print("Visualizing Paper Embeddings")
    print("=" * 60)
    print()

    # Load indexer
    print("[1/4] Loading vector database...")
    indexer = VectorIndexer(
        collection_name="papers",
        persist_directory="data/vector_db"
    )

    # Get all embeddings
    print("[2/4] Fetching all embeddings...")
    collection = indexer.collection

    # Get all items
    results = collection.get(include=['embeddings', 'metadatas'])

    if not results['ids']:
        print("No embeddings found! Run: python scripts/index_papers.py first")
        return

    embeddings = np.array(results['embeddings'])
    metadatas = results['metadatas']

    print(f"      Found {len(embeddings)} chunks")
    print()

    # Group by paper
    print("[3/4] Grouping chunks by paper...")
    paper_embeddings = defaultdict(list)
    paper_titles = {}

    for embedding, metadata in zip(embeddings, metadatas):
        paper_id = metadata['paper_id']
        paper_embeddings[paper_id].append(embedding)
        paper_titles[paper_id] = metadata.get('title', f'Paper {paper_id}')

    # Average embeddings per paper for cleaner visualization
    avg_embeddings = []
    paper_ids = []
    titles = []

    for paper_id, embs in paper_embeddings.items():
        avg_embedding = np.mean(embs, axis=0)
        avg_embeddings.append(avg_embedding)
        paper_ids.append(paper_id)
        titles.append(paper_titles[paper_id][:40])  # Truncate long titles

    avg_embeddings = np.array(avg_embeddings)

    print(f"      Averaged {len(paper_embeddings)} papers")
    print()

    # Apply t-SNE
    print("[4/4] Applying t-SNE dimensionality reduction...")
    print("      (This may take a moment...)")

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(5, len(avg_embeddings) - 1),  # Adjust for small datasets
        max_iter=1000
    )

    embeddings_2d = tsne.fit_transform(avg_embeddings)
    print()

    # Create visualization
    print("Creating visualization...")
    plt.figure(figsize=(14, 10))

    # Plot points
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=paper_ids,
        cmap='tab20',
        s=200,
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )

    # Add labels
    for i, (x, y) in enumerate(embeddings_2d):
        plt.annotate(
            f"{paper_ids[i]}: {titles[i]}",
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )

    plt.title("Paper Embeddings Visualization (t-SNE)", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.colorbar(scatter, label='Paper ID')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_path = Path("visualization_embeddings.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path.absolute()}")
    print()

    # Show plot
    print("Displaying visualization...")
    print("(Close the window to continue)")
    plt.show()

    print()
    print("=" * 60)
    print("Interpretation:")
    print("=" * 60)
    print("Papers that are CLOSE together have similar content.")
    print("Papers that are FAR apart discuss different topics.")
    print()
    print("Look for clusters:")
    print("  - Transformer papers should cluster together")
    print("  - Mamba/State-space models should cluster together")
    print("  - Classic vision papers (HOG, SIFT) might cluster together")
    print()


if __name__ == "__main__":
    visualize_embeddings()
