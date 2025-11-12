"""
Indexer Module

Handles chunking of text and creating vector embeddings for semantic search.
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class TextChunker:
    """Splits text into overlapping chunks for better context preservation."""

    def __init__(self, chunk_size: int = 500, overlap: int = 75):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size of each chunk in tokens (approximate)
            overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, paper_id: int, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks with metadata.

        Args:
            text: Text to chunk
            paper_id: ID of the paper
            metadata: Additional metadata to attach to chunks

        Returns:
            List of chunks with metadata
        """
        # Simple word-based chunking (can be improved with proper tokenization)
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunk_metadata = {
                'paper_id': paper_id,
                'chunk_id': len(chunks),
                'start_word': i,
                'end_word': i + len(chunk_words),
                **(metadata or {})
            }

            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })

        return chunks


class VectorIndexer:
    """Creates and manages vector embeddings using ChromaDB."""

    def __init__(self, collection_name: str = "papers",
                 persist_directory: str = "data/vector_db"):
        """
        Initialize the vector indexer.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Initialize or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Load embedding model
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        print("Embedding model loaded successfully!")

    def index_chunks(self, chunks: List[Dict]):
        """
        Index a list of chunks into the vector database.

        Args:
            chunks: List of chunks with 'text' and 'metadata'
        """
        if not chunks:
            return

        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]

        # Generate unique IDs - include start_word to ensure uniqueness
        ids = [f"{chunk['metadata']['paper_id']}_p{chunk['metadata'].get('page_num', 0)}_c{chunk['metadata']['chunk_id']}_w{chunk['metadata'].get('start_word', 0)}"
               for chunk in chunks]

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )

        print(f"Indexed {len(chunks)} chunks successfully!")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks given a query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching chunks with metadata and similarity scores
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]

        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })

        return formatted_results

    def delete_paper_chunks(self, paper_id: int):
        """Delete all chunks for a specific paper."""
        # Query all chunks for this paper
        results = self.collection.get(
            where={"paper_id": paper_id}
        )

        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks for paper {paper_id}")
