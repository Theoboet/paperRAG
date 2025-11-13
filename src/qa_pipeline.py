"""
Q&A Pipeline

Orchestrates the complete question-answering flow:
1. Retrieve relevant chunks from papers
2. Format context for LLM
3. Generate answer
4. Return result with sources
"""

import time
from typing import Dict, List, Optional
from pathlib import Path

from src.retriever import Retriever
from src.generator import AnswerGenerator
from src.library import PaperLibrary
from src.indexer import VectorIndexer


class QAPipeline:
    """End-to-end question-answering pipeline."""

    def __init__(self,
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 data_dir: str = "data",
                 load_in_8bit: bool = False):
        """
        Initialize the Q&A pipeline.

        Args:
            model_name: HuggingFace model name (TinyLlama or Qwen)
            data_dir: Directory containing papers and database
            load_in_8bit: Use 8-bit quantization for 2x faster inference (default: False)
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit

        # Initialize components
        self.indexer = VectorIndexer(
            collection_name="papers",
            persist_directory=str(self.data_dir / "vector_db")
        )
        self.retriever = Retriever(self.indexer)
        self.library = PaperLibrary(db_path=str(self.data_dir / "papers.db"))

        # Lazy-load generator (heavy operation)
        self.generator = None
        self._model_loaded = False

    def ask(self,
            question: str,
            top_k: int = 3,
            max_tokens: int = 300,
            temperature: float = 0.5,
            verbose: bool = False) -> Dict:
        """
        Ask a question and get an answer with sources.

        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            max_tokens: Maximum tokens in generated answer
            temperature: LLM sampling temperature (0.1-1.0)
            verbose: Print progress messages

        Returns:
            Dictionary with:
                - answer: Generated answer text
                - sources: List of source chunks with metadata
                - time: Generation time in seconds
                - retrieval_scores: Relevance scores for sources
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}\n")

        # Check if papers are indexed
        stats = self.get_stats()
        if stats['paper_count'] == 0:
            return {
                'answer': "No papers found in library. Please add papers first using scripts/upload_paper.py",
                'sources': [],
                'time': 0,
                'retrieval_scores': []
            }

        if stats['indexed_count'] == 0:
            return {
                'answer': "Papers not indexed yet. Please run scripts/index_papers.py first.",
                'sources': [],
                'time': 0,
                'retrieval_scores': []
            }

        # Step 1: Retrieve relevant chunks with variable count
        if verbose:
            print(f"Searching {stats['paper_count']} papers...")

        retrieval_start = time.time()

        # Retrieve more candidates than needed (top_k * 3)
        candidates = self.retriever.retrieve(question, top_k=top_k * 3)

        # Select chunks up to token budget (adaptive)
        results = self._select_chunks_by_tokens(candidates, max_context_tokens=1400, verbose=verbose)

        retrieval_time = time.time() - retrieval_start

        if not results:
            return {
                'answer': "I couldn't find relevant information in your papers for this question.",
                'sources': [],
                'time': retrieval_time,
                'retrieval_scores': []
            }

        if verbose:
            print(f"Selected {len(results)} passages from {len(candidates)} candidates ({retrieval_time:.1f}s)")
            print()

        # Step 2: Format context for LLM
        context = self._format_context(results, verbose=verbose)

        # Step 3: Load model (lazy)
        self._ensure_model_loaded(verbose=verbose)

        # Step 4: Generate answer
        if verbose:
            print("Generating answer...")
            print("(This may take 1-3 minutes on CPU)")
            print()

        gen_start = time.time()
        result = self.generator.generate_answer(
            question=question,
            context=context,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        gen_time = time.time() - gen_start

        total_time = retrieval_time + gen_time

        if verbose:
            print(f"Answer generated ({gen_time:.1f}s)")
            print()

        # Step 5: Extract metadata for sources
        sources = self._enrich_sources(results)

        return {
            'answer': result['answer'],
            'sources': sources,
            'time': total_time,
            'retrieval_time': retrieval_time,
            'generation_time': gen_time,
            'retrieval_scores': [1 - r.get('distance', 0) for r in results]  # Convert distance to similarity
        }

    def _ensure_model_loaded(self, verbose: bool = False):
        """Load the LLM model if not already loaded."""
        if not self._model_loaded:
            if verbose:
                quant_status = " (8-bit quantized)" if self.load_in_8bit else ""
                print(f"Loading language model{quant_status} (first time only)...")
                print()

            self.generator = AnswerGenerator(
                model_name=self.model_name,
                load_in_8bit=self.load_in_8bit
            )
            self.generator.load_model()
            self._model_loaded = True

    def _select_chunks_by_tokens(self, chunks: List[Dict], max_context_tokens: int = 1400, verbose: bool = False) -> List[Dict]:
        """
        Select chunks up to token budget (adaptive chunk count).

        Args:
            chunks: List of candidate chunks from retrieval
            max_context_tokens: Maximum tokens for context (default: 1400)
            verbose: Print selection details

        Returns:
            Selected chunks that fit within token budget
        """
        if not chunks:
            return []

        selected = []
        total_tokens = 0

        for chunk in chunks:
            # Estimate tokens: ~4 characters per token
            chunk_text = chunk.get('text', '')
            estimated_tokens = len(chunk_text) // 4

            # Check if this chunk fits
            if total_tokens + estimated_tokens <= max_context_tokens:
                selected.append(chunk)
                total_tokens += estimated_tokens
            else:
                # Stop if we've hit the limit
                break

        # Always return at least 1 chunk even if it exceeds budget
        # (Better to have long context than no context)
        if not selected and chunks:
            selected = [chunks[0]]
            if verbose:
                print(f"Warning: First chunk ({len(chunks[0]['text']) // 4} tokens) exceeds budget")

        if verbose and selected:
            print(f"Token budget: {total_tokens}/{max_context_tokens} tokens ({len(selected)} chunks)")

        return selected

    def _format_context(self, results: List[Dict], verbose: bool = False) -> str:
        """
        Format retrieved chunks as context for the LLM.

        Args:
            results: List of retrieved chunks with metadata
            verbose: Print context preview

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            # Get paper title from library
            paper_id = result['metadata'].get('paper_id', 'Unknown')
            paper_info = self.library.get_paper(paper_id)
            paper_title = paper_info.get('title', f'Paper {paper_id}') if paper_info else f'Paper {paper_id}'

            # Format chunk
            chunk_text = result['text']
            page_num = result['metadata'].get('page_num', 'Unknown')

            context_parts.append(
                f"[Source {i} - {paper_title}, Page {page_num}]\n{chunk_text}"
            )

        context = "\n\n".join(context_parts)

        if verbose:
            print(f"Context: {len(context)} characters from {len(results)} sources")
            print()

        return context

    def _enrich_sources(self, results: List[Dict]) -> List[Dict]:
        """
        Add paper metadata to source chunks.

        Args:
            results: Retrieved chunks

        Returns:
            Enriched sources with paper titles, authors, etc.
        """
        sources = []

        for result in results:
            paper_id = result['metadata'].get('paper_id', 'Unknown')
            paper_info = self.library.get_paper(paper_id)

            # Calculate relevance score from distance (lower distance = higher relevance)
            distance = result.get('distance', 1.0)
            relevance = 1 - distance  # Convert to similarity

            if paper_info:
                source = {
                    'paper_id': paper_id,
                    'paper_title': paper_info.get('title', 'Unknown'),
                    'authors': paper_info.get('authors', []),
                    'year': paper_info.get('year', 'Unknown'),
                    'page': result['metadata'].get('page_num', 'Unknown'),
                    'text_preview': result['text'][:200] + '...' if len(result['text']) > 200 else result['text'],
                    'relevance_score': relevance
                }
            else:
                source = {
                    'paper_id': paper_id,
                    'paper_title': f'Paper {paper_id}',
                    'page': result['metadata'].get('page_num', 'Unknown'),
                    'text_preview': result['text'][:200] + '...',
                    'relevance_score': relevance
                }

            sources.append(source)

        return sources

    def get_stats(self) -> Dict:
        """
        Get library and indexing statistics.

        Returns:
            Dictionary with paper count, index status, etc.
        """
        papers = self.library.list_papers()

        # Check if ChromaDB exists and has data
        indexed = False
        try:
            # Try a dummy search to see if index exists
            self.retriever.retrieve("test", top_k=1)
            indexed = True
        except:
            indexed = False

        return {
            'paper_count': len(papers),
            'indexed_count': len(papers) if indexed else 0,
            'model_name': self.model_name,
            'model_loaded': self._model_loaded
        }

    def unload_model(self):
        """Unload the LLM to free memory."""
        if self._model_loaded and self.generator:
            self.generator.unload_model()
            self._model_loaded = False
            print("Model unloaded from memory.")
