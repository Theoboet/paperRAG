"""
Retriever Module

Handles semantic search and retrieval of relevant text passages.
"""

from typing import List, Dict


class Retriever:
    """Retrieves relevant text passages for a given query."""

    def __init__(self, indexer):
        """
        Initialize the retriever.

        Args:
            indexer: VectorIndexer instance
        """
        self.indexer = indexer

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant chunks with metadata
        """
        results = self.indexer.search(query, top_k=top_k)
        return results

    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            paper_id = chunk['metadata'].get('paper_id', 'Unknown')
            page_num = chunk['metadata'].get('page_num', 'Unknown')

            context_parts.append(
                f"[Chunk {i} - Paper ID: {paper_id}, Page: {page_num}]\n"
                f"{chunk['text']}\n"
            )

        return "\n".join(context_parts)
