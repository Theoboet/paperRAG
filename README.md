# paperRAG

An offline RAG (Retrieval-Augmented Generation) system for PDF documents. Ask questions about your paper library and get answers with source citations, all running locally on your machine without internet connection.

## Two Ways to Use paperRAG

### Option A: Try with Example Computer Vision Papers

Get started quickly with pre-loaded CV papers (ResNet, ViT, YOLO):

```bash
git clone https://github.com/Theoboet/paperRAG
cd paperRAG
git checkout example-cv-papers  # Switch to branch with example papers
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/index_papers.py
python scripts/ask.py "How does Vision Transformer work?"
```

### Option B: Use Your Own Papers (Any Domain)

Works with ANY PDF collection (research papers, legal documents, reports, textbooks):

```bash
git clone https://github.com/Theoboet/paperRAG
cd paperRAG
# Stay on main branch
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/upload_paper.py path/to/your/paper.pdf
python scripts/index_papers.py
python scripts/ask.py "Your question"
```

## Motivation

Reading computer vision research papers is challenging. Papers are dense with technical jargon, complex architectures, and mathematical formulations. Researchers often need to:

1. Cross-reference multiple papers to understand a concept
2. Find specific information buried in 10+ page documents
3. Compare approaches across different papers
4. Understand technical details without re-reading entire papers

Existing solutions like ChatGPT and Claude have limitations:

- **Context Window Limits**: Cannot process large document collections at once. Even with 128k token context windows, you can't upload dozens of papers simultaneously. This forces you to repeatedly upload subsets of papers, making cross-paper comparisons difficult.
- **Hallucinations**: When you do upload documents, models may generate plausible-sounding information that isn't actually in the papers, especially when context limits force you to truncate content.
- **No Incremental Knowledge**: Each conversation starts fresh. You can't build a persistent library that grows over time.
- **Internet Dependency**: Require active connection and API access.
- **Privacy Concerns**: Your papers are sent to external servers.
- **No Source Citations**: Don't provide specific page numbers or reliable references to source material.

paperRAG solves these problems with a RAG (Retrieval-Augmented Generation) approach: it creates a local, searchable knowledge base from unlimited PDFs, retrieves only the most relevant passages for each question (fitting within context limits), and generates answers with precise source citations.

## Features

- **Unlimited Document Library**: Index hundreds of papers without context window limits. RAG retrieves only the most relevant passages for each question, so you're never constrained by model context size.
- **No Hallucinations from Truncation**: Unlike uploading truncated documents to ChatGPT, RAG ensures the model only sees complete, relevant passages retrieved from your full library.
- **Fully Offline**: All processing and inference runs locally. No API keys, no internet required. Your papers stay private.
- **Semantic Search**: Uses sentence embeddings (all-mpnet-base-v2) to find relevant passages across your entire paper library, no matter how large.
- **Precise Source Citations**: Every answer includes references to specific papers and page numbers, eliminating hallucination concerns.
- **Persistent Knowledge Base**: Build your library incrementally over time. Each new paper enriches your searchable knowledge without re-processing.
- **Adaptive Context**: Dynamically selects the most relevant chunks to fit within token budget, maximizing information density.
- **Multiple Models**: Choose between TinyLlama (fast, 1-2 min) or Qwen (high quality, GPU recommended).
- **Quantization Support**: Optional 8-bit quantization for 2x faster inference with minimal quality loss.
- **Interactive Mode**: Ask multiple questions in a single session without reloading the model.

## System Architecture

paperRAG implements a classic RAG pipeline with optimizations for local execution:

### 1. Document Processing

**PDF Parsing** (src/parser.py):
- Uses PyMuPDF to extract text and metadata from research papers
- Extracts title, authors, publication year from document metadata
- Page-level text extraction with position tracking

**Paper Library** (src/library.py):
- SQLite database stores paper metadata (title, authors, year, file path)
- Each paper receives a unique ID for tracking
- Supports CRUD operations and paper listing

### 2. Text Chunking and Embedding

**Chunking Strategy** (src/indexer.py):
- Splits papers into 500-token chunks with 75-token overlap
- Overlap preserves context at chunk boundaries
- Maintains metadata (paper_id, page_num, chunk_id) for each chunk

**Embedding Model**:
- all-mpnet-base-v2 from sentence-transformers
- 768-dimensional dense vectors
- Optimized for semantic similarity search
- Model cached locally after first download (420MB)

**Vector Storage**:
- ChromaDB persistent vector database
- Cosine similarity for retrieval
- Stores embeddings, text, and metadata together
- Incremental updates: only new papers are embedded

### 3. Retrieval

**Semantic Search** (src/retriever.py):
- Converts user question to 768-dim embedding vector
- Performs cosine similarity search in ChromaDB
- Returns top-k most relevant chunks with relevance scores

**Adaptive Chunk Selection** (src/qa_pipeline.py):
- Retrieves 3x requested chunks as candidates
- Greedily selects chunks until token budget is filled (1400 tokens for context)
- Prioritizes most relevant chunks while respecting model's context window
- Prevents token overflow on broad questions with large chunks

### 4. Answer Generation

**Language Models** (src/generator.py):
- TinyLlama-1.1B-Chat (default): 1.1B parameters, 1-2 min on CPU
- Qwen2.5-7B-Instruct (optional): 7B parameters, better quality, GPU recommended
- Models auto-download from HuggingFace on first use
- Cached in ~/.cache/huggingface/ for reuse

**Prompt Engineering**:
- System prompt: Research assistant that explains concepts clearly
- Context: Formatted chunks with paper titles and page numbers
- Question: User's original query
- Temperature: 0.5 (balanced between creativity and focus)

**Optimizations**:
- Lazy loading: Model loads only when needed
- 8-bit quantization: Optional 2x speedup via bitsandbytes
- Context truncation: Adapts chunk count to fit token budget

### 5. Q&A Pipeline

**End-to-End Flow** (src/qa_pipeline.py):

1. **Validate library**: Check if papers exist and are indexed
2. **Retrieve candidates**: Get 3x top_k most relevant chunks
3. **Select context**: Pack chunks into 1400-token budget
4. **Format context**: Add paper titles and page numbers to each chunk
5. **Load model**: Lazy-load LLM if not already in memory
6. **Generate answer**: Pass context + question to LLM
7. **Extract sources**: Enrich chunk metadata with paper details
8. **Return result**: Answer + sources + timing information

## Installation

### Requirements

- Python 3.9+
- 10GB+ free disk space (for models and vector database)
- 16GB RAM recommended (8GB minimum)
- GPU optional but recommended for Qwen model

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/paperRAG.git
cd paperRAG
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

This installs:
- PyMuPDF (PDF parsing)
- sentence-transformers (embeddings)
- ChromaDB (vector database)
- PyTorch (deep learning backend)
- transformers (LLM inference)
- bitsandbytes (quantization)
- accelerate (optimization)

On first run, the system will automatically download:
- all-mpnet-base-v2 embedding model (420MB)
- TinyLlama-1.1B-Chat language model (4.4GB)

## Quick Start

### 1. Add Papers to Your Library

```bash
# Add a single paper
python scripts/upload_paper.py path/to/paper.pdf

# Add multiple papers
python scripts/upload_paper.py paper1.pdf paper2.pdf paper3.pdf

# List all papers in library
python scripts/list_papers.py
```

Papers are stored in `data/papers/` and metadata is saved in `data/papers.db`.

### 2. Index Your Papers

```bash
python scripts/index_papers.py
```

This process:
- Chunks each paper into 500-token segments
- Generates embeddings for each chunk
- Stores vectors in ChromaDB (`data/vector_db/`)
- Takes ~30 seconds per paper

Note: Re-running this script only indexes new papers. Existing papers are skipped.

### 3. Ask Questions

**Single Question Mode**:
```bash
python scripts/ask.py "How does Vision Transformer work?"
```

**Interactive Mode**:
```bash
python scripts/ask.py --interactive
```

Type your questions and press Enter. Type 'quit' to exit.

**With Options**:
```bash
# Use Qwen model (slower, higher quality)
python scripts/ask.py "What is attention?" --model qwen

# Use 8-bit quantization (2x faster)
python scripts/ask.py "What is attention?" --quant

# Verbose output with timing details
python scripts/ask.py "What is attention?" --verbose

# Retrieve more chunks for context
python scripts/ask.py "What is attention?" --top-k 5

# Control answer length
python scripts/ask.py "What is attention?" --max-tokens 500
```

## Usage Examples

### Example 1: Understanding a Specific Architecture

```bash
$ python scripts/ask.py "How does Vision Transformer work?"

======================================================================
ANSWER
======================================================================

Vision Transformer (ViT) applies the Transformer architecture directly
to image patches. The image is split into fixed-size patches (typically
16x16 pixels), each patch is linearly embedded, and positional embeddings
are added. These patch embeddings are then fed into a standard Transformer
encoder. A special classification token is prepended to the sequence, and
its final hidden state is used for image classification.

The key innovation is treating images as sequences of patches rather than
pixels, allowing Transformers to scale to large datasets. ViT achieves
excellent results when pre-trained on large datasets like ImageNet-21k.

======================================================================
SOURCES
======================================================================

1. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
   Authors: Dosovitskiy et al.
   Year: 2021
   Page: 3

2. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
   Authors: Dosovitskiy et al.
   Year: 2021
   Page: 4

======================================================================
Time: 156.3s
======================================================================
```

### Example 2: Comparing Approaches

```bash
$ python scripts/ask.py "What are the differences between ResNet and Vision Transformer?" --top-k 5

# Retrieves passages from both ResNet and ViT papers
# Generates comparison based on retrieved context
```

### Example 3: Understanding Technical Details

```bash
$ python scripts/ask.py "What is self-attention and how is it computed?" --model qwen

# Uses higher-quality Qwen model for detailed technical explanation
```

## Configuration Options

### Model Selection

**TinyLlama (Default)**:
- 1.1B parameters
- 1-2 minutes per answer on CPU
- Good quality for most questions
- 4.4GB download

**Qwen (Optional)**:
- 7B parameters
- 14-20 minutes per answer on CPU (recommend GPU)
- Higher quality, more detailed answers
- 15GB download

Use with: `--model qwen`

### Quantization

8-bit quantization reduces model precision from 32-bit to 8-bit:
- 2x faster inference
- Slight quality degradation (usually acceptable)
- Same model download size

Use with: `--quant` or `--quantized`

### Retrieval Parameters

**--top-k N**: Number of chunk candidates to retrieve (default: 3)
- Higher values provide more context but may exceed token budget
- System automatically selects chunks that fit within 1400-token limit
- Recommended: 3-5 for specific questions, 5-10 for broad questions

**--max-tokens N**: Maximum length of generated answer (default: 300)
- Longer answers take more time to generate
- Must fit within model's total context window (2048 tokens for TinyLlama)
- Recommended: 200-400

**--temperature T**: Sampling temperature 0.0-1.0 (default: 0.5)
- Lower values (0.1-0.3): More focused, deterministic answers
- Higher values (0.7-1.0): More creative, diverse answers
- Recommended: 0.5 for balanced results

### Output Options

**--verbose**: Show detailed timing and retrieval information
**--interactive**: Enter interactive mode for multiple questions

## Technical Details

### Token Budget Management

TinyLlama has a 2048 token context window. Token allocation:

- System prompt: ~150 tokens
- Context chunks: up to 1400 tokens (adaptive)
- Question: ~50 tokens
- Answer generation: ~300 tokens (configurable)

The system dynamically selects chunks to maximize relevant information while staying within budget:

1. Retrieve 3x candidates (e.g., top_k=3 retrieves 9 chunks)
2. Sort candidates by relevance (from vector search)
3. Greedily pack chunks until 1400-token budget is reached
4. Always include at least 1 chunk (even if oversized)

This approach handles both narrow questions (many small chunks fit) and broad questions (fewer large chunks fit).

### Embedding Model

all-mpnet-base-v2 specifications:
- Architecture: Sentence-BERT with MPNet backbone
- Dimensions: 768
- Context window: 384 tokens
- Training: 1B+ sentence pairs
- Performance: 63.3 on semantic textual similarity benchmarks

### Vector Search

ChromaDB configuration:
- Distance metric: Cosine similarity
- Index: HNSW (Hierarchical Navigable Small World)
- Persistence: Disk-backed storage in data/vector_db/
- Metadata filtering: Supported but not currently used

Retrieval process:
1. Encode query with all-mpnet-base-v2
2. Compute cosine similarity with all chunk embeddings
3. Return top-k results with distance scores
4. Distance to similarity: similarity = 1 - distance

### Model Inference

Inference pipeline (src/generator.py):
1. Format messages with chat template (role: system/user/assistant)
2. Tokenize input text
3. Generate tokens autoregressively
4. Apply temperature sampling
5. Decode output tokens to text
6. Extract assistant response

Optimization techniques:
- bitsandbytes: 8-bit quantization for faster inference
- torch.no_grad(): Disable gradient computation
- Lazy loading: Load model only when needed
- Model persistence: Keep model in memory across questions (interactive mode)

### Prompt Design

System prompt emphasizes:
- Clear, comprehensive explanations
- Simple language with technical term definitions
- Logical structure (overview, details, results)
- Honesty when context lacks information
- Focus on helpfulness

Context format:
```
[Source 1 - Paper Title, Page N]
Text from chunk 1...

[Source 2 - Paper Title, Page M]
Text from chunk 2...
```

This format helps the model understand source boundaries and enables citation generation.

## Performance Benchmarks

Tested on Intel Core i7 CPU (no GPU), 16GB RAM:

### TinyLlama (Default)

| Question Type | Chunks Selected | Generation Time | Total Time | Quality |
|--------------|----------------|-----------------|------------|---------|
| Specific ("How does ViT work?") | 3 | 120s | 156s | Good |
| Broad ("What is computer vision?") | 2 | 90s | 118s | Good |
| Technical ("What is self-attention?") | 3 | 135s | 162s | Good |

### TinyLlama with 8-bit Quantization

| Question Type | Generation Time | Speedup | Quality Loss |
|--------------|-----------------|---------|--------------|
| Specific | 60s | 2x | Minimal |
| Broad | 45s | 2x | Minimal |
| Technical | 68s | 2x | Minimal |

### Qwen (7B, CPU)

| Question Type | Generation Time | Total Time | Quality |
|--------------|-----------------|------------|---------|
| Specific | 840s (14 min) | 852s | Excellent |
| Broad | 720s (12 min) | 735s | Excellent |
| Technical | 960s (16 min) | 982s | Excellent |

Note: Qwen performance improves dramatically with GPU (10-20x faster).

## Project Structure

```
paperRAG/
├── src/                          # Core modules
│   ├── __init__.py
│   ├── parser.py                # PDF parsing with PyMuPDF
│   ├── library.py               # Paper metadata management (SQLite)
│   ├── indexer.py               # Text chunking and embedding
│   ├── retriever.py             # Semantic search
│   ├── generator.py             # LLM answer generation
│   └── qa_pipeline.py           # End-to-end Q&A orchestration
├── scripts/                      # User-facing scripts
│   ├── ask.py                   # Main Q&A interface
│   ├── upload_paper.py          # Add papers to library
│   ├── index_papers.py          # Generate embeddings
│   ├── list_papers.py           # List library contents
│   └── visualize_embeddings.py  # t-SNE visualization utility
├── data/                         # User data (not in git)
│   ├── papers/                  # PDF storage
│   ├── papers.db                # SQLite database
│   └── vector_db/               # ChromaDB storage
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── QUICKSTART.md                # Quick setup guide
└── .gitignore                   # Git ignore rules
```

## Adding New Papers

Papers can be added at any time without re-indexing existing papers:

```bash
# Add new paper
python scripts/upload_paper.py new_paper.pdf

# Index only the new paper
python scripts/index_papers.py

# Start asking questions
python scripts/ask.py "What does the new paper say about X?"
```

The indexing script automatically detects which papers are already indexed and only processes new additions.

## Troubleshooting

### Model Download Issues

If model download fails or is interrupted:
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0
python scripts/ask.py "test question"
```

### Out of Memory Errors

If you encounter OOM errors:
1. Use 8-bit quantization: `--quant`
2. Reduce max_tokens: `--max-tokens 200`
3. Reduce top_k: `--top-k 2`
4. Close other applications
5. Use TinyLlama instead of Qwen

### Slow Generation

For faster answers:
1. Use 8-bit quantization: `--quant` (2x speedup)
2. Use TinyLlama (default, 12x faster than Qwen)
3. Reduce max_tokens: `--max-tokens 200`
4. Use GPU if available (10-20x speedup)

### Token Overflow Errors

If you see "Token sequence length exceeds maximum":
1. System automatically handles this with adaptive chunk selection
2. If still occurring, reduce top_k: `--top-k 2`
3. Avoid extremely broad questions

### Empty or Poor Quality Answers

If answers are not helpful:
1. Verify papers are indexed: `python scripts/list_papers.py`
2. Check papers contain relevant information
3. Try increasing top_k: `--top-k 5`
4. Try Qwen model for better quality: `--model qwen`
5. Rephrase question to be more specific

## Limitations

1. **Speed**: CPU-only inference is slow (1-2 min per answer with TinyLlama)
2. **Context Window**: Limited to 2048 tokens (TinyLlama), restricts how much context can be used
3. **Model Size**: TinyLlama is small (1.1B parameters), may not handle very complex questions
4. **Domain**: Optimized for computer vision papers, may work less well for other domains
5. **PDF Quality**: Relies on text extraction; scanned PDFs without OCR will not work
6. **No Cross-Paper Reasoning**: Retrieves chunks independently; does not explicitly link concepts across papers

## Future Improvements

Potential enhancements:
1. **GPU Support**: Automatic GPU detection and usage for 10-20x speedup
2. **Larger Context**: Support models with 4k-8k context windows
3. **Better Chunking**: Semantic chunking based on section boundaries
4. **Query Expansion**: Automatically expand queries with synonyms and related terms
5. **Re-ranking**: Two-stage retrieval with cross-encoder re-ranking
6. **Caching**: Cache frequently asked questions
7. **Web UI**: Gradio or Streamlit interface for easier interaction
8. **Multi-modal**: Support figures and tables from PDFs
9. **Graph RAG**: Build knowledge graph linking concepts across papers
10. **Incremental Learning**: Fine-tune embeddings on user's paper corpus

## Contributing

Contributions are welcome! Areas for improvement:
- Faster inference optimizations
- Better chunking strategies
- Additional model support
- UI development
- Bug fixes and testing

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Sentence-Transformers for embedding models
- ChromaDB for vector storage
- HuggingFace for LLM hosting
- TinyLlama and Qwen teams for open-source models
- PyMuPDF for PDF parsing

## Citation

If you use paperRAG in your research, please cite:

```bibtex
@software{paperrag2024,
  title={paperRAG: Offline RAG System for Computer Vision Papers},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/paperRAG}
}
```
