# paperRAG - Offline Assistant for Reading Computer Vision Papers

A local application that helps you understand computer vision papers offline with citations and semantic search.

## Project Status

Currently implementing **Phase 1: Foundation** (Step 1.1 completed)

### Completed:
- [x] Project structure setup
- [x] Virtual environment created
- [x] Core dependencies installed (PyMuPDF, ChromaDB, sentence-transformers, etc.)
- [x] Initial Python modules created:
  - `pdf_parser.py` - PDF text extraction
  - `library.py` - Paper management with SQLite
  - `indexer.py` - Text chunking and vector embeddings
  - `retriever.py` - Semantic search
  - `generator.py` - Answer generation (placeholder for LLM integration)

## Setup

1. Activate the virtual environment:
   ```bash
   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

2. (Optional) Verify installation:
   ```bash
   python -c "import pymupdf, chromadb, sentence_transformers; print('All packages imported successfully!')"
   ```

## Quick Start - Upload Your First Paper

```bash
# Upload a paper
python upload_paper.py path/to/paper.pdf

# List all papers
python list_papers.py
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed instructions on where to get papers and how to use the system.**

### Where to Place PDFs:
- **Option 1**: Use the upload script (recommended): `python upload_paper.py paper.pdf`
- **Option 2**: Manually place PDFs in: `data/papers/` directory

## Project Structure

```
paperRAG/
├── src/
│   ├── __init__.py
│   ├── pdf_parser.py      # PDF text extraction
│   ├── library.py          # Paper database management
│   ├── indexer.py          # Text chunking & embeddings
│   ├── retriever.py        # Semantic search
│   └── generator.py        # Answer generation
├── models/                 # LLM models (to be added)
├── data/
│   ├── papers/            # Uploaded PDF papers
│   └── vector_db/         # ChromaDB vector storage
├── ui/                     # User interface (Phase 4)
├── venv/                   # Virtual environment
├── requirements.txt
└── README.md
```

## Next Steps

As per the implementation plan:

**Step 1.2: PDF Parsing Module** (Next)
- Test PDF extraction with real CV papers
- Improve metadata extraction
- Add error handling

**Step 1.3: Library Management**
- Test CRUD operations
- Add search functionality

**Phase 2: RAG Core (Weeks 3-4)**
- Implement chunking strategy
- Download and test embedding model
- Set up ChromaDB
- Integrate local LLM

## Hardware Requirements

- **RAM**: 16GB recommended (8GB for embeddings, 8GB for LLM)
- **Storage**: 10GB+ for models and vector database
- **GPU**: Optional but recommended for faster LLM inference

## Technology Stack

- **PDF Processing**: PyMuPDF (fitz)
- **Embeddings**: sentence-transformers (all-mpnet-base-v2)
- **Vector DB**: ChromaDB
- **Database**: SQLite
- **LLM**: To be added (llama-cpp-python with quantized model)

## Note on LLM Integration

The LLM integration (`llama-cpp-python`) was skipped in initial setup because it requires C++ build tools. This will be added in Phase 2.3 when we integrate the local language model.

Alternative options:
1. Install C++ Build Tools and reinstall llama-cpp-python
2. Use transformers library with CPU inference
3. Download pre-built wheels for llama-cpp-python
