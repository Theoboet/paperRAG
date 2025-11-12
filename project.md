# Project: Offline Assistant for Reading Computer Vision Papers

## Goal
Build a local app that helps you **understand computer vision papers offline**.  
It lets you ask questions, get **answers with exact paper citations**, and receive **concise summaries**, all without an internet connection.

---

## Problem
Computer vision papers are long, dense, and full of technical terms.  
When traveling or offline, it’s hard to look things up or compare papers.  
You need a tool that makes reading **faster** and **easier to understand**, even without the web.

---

## Solution
A **reading assistant app** that works entirely on your laptop:
- You upload your PDF papers.  
- The app “reads” them and builds an internal map of their contents.  
- When you ask a question, it finds the most relevant passages and uses them to answer, **always citing the source (paper and page)**.  

It doesn’t invent facts — it only uses the text from your files.

---

## Core Components

### 1. Library  
Stores all your papers in one place with metadata (title, authors, year).  
Lets you browse, filter, and manage your collection.

### 2. Indexing  
The app reads each PDF, splits it into small chunks of text, and turns them into vector representations.  
This builds a **semantic map** of your library so it knows where every concept appears.

### 3. Retrieval  
When you ask a question, the system searches this semantic map and selects the most relevant text fragments.  
It’s the logical part that knows **where to look**.

### 4. Generation  
A small **local language model** reads the retrieved fragments and produces a clear answer.  
It **always includes citations** like `(Paper: Smith2023, p. 7)` or `(Paper: ViT, Table 2, p. 9)`.

### 5. Interface  
The user-facing window.  
You can:
- Upload new papers  
- View your library  
- Ask questions and read answers  
- Click citations to open the corresponding PDF page  
- See automatic paper summaries  

---

## Citations System
Each answer clearly shows **where the information came from**:
- **Per sentence** for numbers or metrics.  
- **Per paragraph** for broader summaries.  
- **Multiple citations** when combining sources.  
- If no information is found, it replies: *“Not found in your papers.”*

Example:
- Introduces a loss focused on false negatives. (Paper: *XYZ 2023*, p. 5)  
- Trained on COCO for 300 epochs, batch 1024, cosine LR. (Paper: *XYZ 2023*, p. 7)  
- Outperforms DeiT by +1.6 top-1 on ImageNet-1k. (Paper: *XYZ 2023*, Tab. 2, p. 9)

---

## Workflow

1. **Import** → Add PDFs to your local library.  
2. **Index** → The app reads and maps them.  
3. **Ask** → You type a question.  
4. **Retrieve** → It finds the best text fragments.  
5. **Generate** → The model composes a clear, cited answer.  
6. **Review** → You can open the cited pages instantly.

---

## Benefits
- Works fully offline.  
- Keeps all data private on your device.  
- Saves hours of reading time.  
- Provides accurate, verifiable answers.  
- Makes complex papers understandable through guided Q&A.

---

## Identity
It's not a search engine or ChatGPT clone — it's a **personal research companion**.
Private, portable, and built for serious students and researchers who want clarity, speed, and autonomy.

---

## Technical Analysis & Considerations

### Strengths of This Approach
- **Privacy-first**: All data stays local - crucial for unpublished/proprietary research
- **Offline operation**: Perfect for travel, planes, areas with poor connectivity
- **Verifiable**: Citations ensure trustworthiness and allow fact-checking
- **Focused domain**: Computer vision papers are a well-defined niche with consistent structure

### Critical Challenges & Things to Watch Out For

#### 1. **PDF Parsing Quality**
- **Problem**: Academic PDFs have complex layouts (multi-column, equations, figures, tables)
- **Risk**: Poor extraction = garbled text = wrong answers
- **Mitigation**: Need robust parsing that handles LaTeX equations, preserves table structure, and correctly orders multi-column text
- **Tools to consider**: PyMuPDF, pdfplumber, or specialized academic PDF parsers

#### 2. **Model Selection & Performance**
- **Challenge**: Local LLMs need to be small enough to run on laptops but smart enough to understand technical content
- **Trade-offs**:
  - Small models (1-3B params): Fast but may struggle with complex technical reasoning
  - Medium models (7-13B): Better understanding but require 8-16GB VRAM
  - Quantized models: Smaller memory footprint but potential quality loss
- **Recommendation**: Start with quantized Llama 3.1 8B or Mistral 7B, measure quality

#### 3. **Citation Accuracy**
- **Critical issue**: The model MUST NOT hallucinate citations
- **Risk**: If the model invents page numbers or misattributes quotes, the tool becomes unreliable
- **Solution**:
  - Strict prompt engineering to force citation from context only
  - Post-processing to verify citations match retrieved chunks
  - Consider a separate citation extraction step before generation

#### 4. **Chunking Strategy**
- **Problem**: How to split papers while preserving context?
- **Considerations**:
  - Too small chunks (200-300 tokens): Lose context, fragment ideas
  - Too large chunks (1000+ tokens): Dilute relevance, waste context window
  - **Recommendation**: 400-600 tokens with 50-100 token overlap, respect paragraph boundaries
  - Special handling for tables and equations (keep intact)

#### 5. **Embedding Model Selection**
- **Challenge**: Need embeddings that understand scientific/technical language
- **Options**:
  - General models (sentence-transformers): Fast but may miss domain nuances
  - Scientific models (SciBERT, SPECTER): Better domain understanding
  - **Recommendation**: Start with `all-mpnet-base-v2`, compare with `allenai/specter2`

#### 6. **Hardware Requirements**
- **Realistic minimums**:
  - RAM: 16GB (8GB embeddings/vectors, 8GB for LLM)
  - Storage: 10GB+ for models, vectors scale with library size
  - CPU: Works but slow; GPU recommended for LLM inference
- **User expectation management**: First query may be slow (model loading)

#### 7. **Vector Database Scaling**
- **Problem**: Large libraries (100+ papers) = millions of vectors
- **Trade-off**: Accuracy vs speed vs memory
- **Options**:
  - FAISS: Fast, mature, good for offline
  - ChromaDB: Easier to use, better DX
  - **Recommendation**: ChromaDB for MVP, FAISS if performance becomes issue

#### 8. **Multi-Document Context**
- **Challenge**: Questions spanning multiple papers ("How does X compare to Y?")
- **Complexity**: Need to retrieve from multiple sources and synthesize
- **Solution**: Retrieve top K chunks regardless of source paper, let model synthesize

### Potential Pitfalls

1. **Scope creep**: Don't try to build a full PDF reader/annotator - focus on Q&A
2. **Perfectionism on parsing**: 90% accuracy on PDF extraction is better than spending months on edge cases
3. **Over-engineering RAG**: Start simple (naive retrieval), optimize later
4. **Ignoring UX**: If queries take >30s, users will abandon it
5. **Not testing with real papers**: Mock data ≠ actual dense CV papers with complex figures

### Security & Safety Concerns

1. **Prompt injection**: If you later add shared libraries, malicious PDFs could contain text that manipulates prompts
2. **Resource exhaustion**: Large PDFs could DoS the system (limit file sizes)
3. **Model safety**: Local models bypass API safety filters - acceptable for research tool

---

## Detailed Implementation Plan

### Phase 1: Foundation (Week 1-2)

#### Step 1.1: Environment Setup
- Set up Python project with virtual environment
- Install core dependencies:
  - `pymupdf` or `pdfplumber` for PDF parsing
  - `sentence-transformers` for embeddings
  - `chromadb` for vector storage
  - `llama-cpp-python` or `transformers` for LLM inference
- Create basic project structure:
  ```
  paperRAG/
  ├── src/
  │   ├── pdf_parser.py
  │   ├── indexer.py
  │   ├── retriever.py
  │   ├── generator.py
  │   └── library.py
  ├── models/
  ├── data/
  │   ├── papers/
  │   └── vector_db/
  └── ui/
  ```

#### Step 1.2: PDF Parsing Module
- Implement PDF text extraction with metadata
- Extract: title, authors, year (from first page patterns)
- Handle: multi-column layouts, preserve section headers
- Test with 5-10 real CV papers (ViT, ResNet, YOLO papers)
- **Deliverable**: Function that returns structured text + metadata

#### Step 1.3: Library Management
- SQLite database for paper metadata
- Schema: `papers(id, title, authors, year, file_path, indexed_date)`
- Basic CRUD operations
- **Deliverable**: Can add/remove/list papers

### Phase 2: RAG Core (Week 3-4)

#### Step 2.1: Chunking & Embedding
- Implement text chunking (start with 500 tokens, 75 token overlap)
- Store chunk metadata: `(paper_id, page_num, chunk_id, text)`
- Download embedding model (all-mpnet-base-v2)
- Generate embeddings for all chunks
- **Deliverable**: Papers → vector representations in ChromaDB

#### Step 2.2: Retrieval System
- Implement semantic search over vectors
- Query embedding → top K similar chunks (K=5-10)
- Include metadata in results (paper title, page, chunk text)
- Test: query "what is attention mechanism" should retrieve relevant chunks
- **Deliverable**: Question → relevant text passages

#### Step 2.3: LLM Integration
- Download quantized model (e.g., Llama-3.1-8B-Instruct-Q4_K_M.gguf)
- Set up llama.cpp or transformers for inference
- Test model loads and generates text
- Measure: loading time, inference speed, memory usage
- **Deliverable**: Local LLM that can generate text

### Phase 3: Generation & Citations (Week 5-6)

#### Step 3.1: Prompt Engineering
- Design system prompt:
  - Role: research assistant analyzing academic papers
  - Rules: cite every claim, use provided context only, say "not found" if no info
  - Format: (Paper: Title, p. X) or (Paper: Title, Table Y, p. X)
- Design user prompt template:
  ```
  Context:
  [Chunk 1 from Paper A, Page 3]
  [Chunk 2 from Paper B, Page 7]

  Question: {user_question}

  Answer based only on the context above. Cite sources for each claim.
  ```
- **Deliverable**: Prompt template that encourages citations

#### Step 3.2: Citation Extraction & Validation
- Post-process LLM output to extract citation markers
- Validate: each citation corresponds to a real retrieved chunk
- Flag/remove hallucinated citations
- Format citations consistently
- **Deliverable**: Reliable citation system

#### Step 3.3: Answer Generation Pipeline
- Chain: Question → Retrieve → Format context → LLM → Validate citations → Return
- Handle edge cases:
  - No relevant chunks found → "Not found in your papers"
  - Multiple papers have info → synthesize with multiple citations
- **Deliverable**: End-to-end Q&A with citations

### Phase 4: Interface (Week 7-8)

#### Step 4.1: Choose UI Framework
- **Options**:
  - Gradio: Fastest MVP, web-based, easy to deploy
  - Streamlit: Similar to Gradio, nice for data apps
  - PyQt/Tkinter: Native desktop app, more work
  - Electron + Python backend: Best UX, most complex
- **Recommendation**: Start with Gradio for MVP

#### Step 4.2: Build Core UI
- **Library View**: Table of papers with metadata, upload button
- **Chat Interface**: Question input, streaming answer output
- **Citations Display**: Clickable citations that show source context
- Progress indicators for indexing/generation
- **Deliverable**: Working UI that connects all components

#### Step 4.3: PDF Viewer Integration
- Clicking citation opens PDF to exact page
- **Options**:
  - External viewer (system default PDF app)
  - Embedded viewer (pdf.js if web-based, or PyMuPDF render if desktop)
- **Deliverable**: Can navigate from answer to source

### Phase 5: Polish & Optimization (Week 9-10)

#### Step 5.1: Performance Optimization
- Cache: embedding model, LLM in memory (don't reload per query)
- Batch operations: index multiple papers at once
- Async where possible: PDF parsing, embedding generation
- Add loading indicators for slow operations

#### Step 5.2: Quality Improvements
- Experiment with chunk sizes (test 300, 500, 700 tokens)
- Try different embedding models, measure retrieval quality
- Adjust retrieval K and LLM context window
- Test with diverse questions (facts, comparisons, methodology questions)

#### Step 5.3: Auto-Summarization
- Add feature: automatically generate paper summary on upload
- Prompt: "Summarize this paper in 3-4 bullet points: contribution, method, results"
- Store summaries in database
- Display in library view

#### Step 5.4: Error Handling & Edge Cases
- Corrupted PDFs → graceful error message
- Very large PDFs (>100 pages) → warn about indexing time
- Extremely long questions → truncate or warn
- Model out of memory → reduce context or batch size

### Phase 6: Testing & Refinement (Week 11-12)

#### Step 6.1: User Testing
- Test with real researchers/students
- Gather feedback on:
  - Answer quality and relevance
  - Citation accuracy
  - Speed and responsiveness
  - UI/UX pain points

#### Step 6.2: Quality Metrics
- Measure retrieval precision: are retrieved chunks relevant?
- Measure citation accuracy: manually verify 50 citations
- Measure answer quality: subjective but get user ratings

#### Step 6.3: Documentation
- README with setup instructions
- Model download guide
- Hardware requirements
- Example questions and workflows
- Troubleshooting guide

### Optional Enhancements (Post-MVP)

1. **Figures & Tables**: Extract and index figure captions, table contents
2. **Paper Comparison**: Built-in prompts for "Compare X and Y"
3. **Export**: Save Q&A sessions for notes
4. **Collections**: Organize papers into topics
5. **Batch Questions**: Ask multiple questions at once
6. **Citation Graph**: Visualize which papers cite each other
7. **Highlighting**: Highlight exact sentences in PDF viewer

---

## Technology Stack Recommendation

### Core Stack
- **Language**: Python 3.10+
- **PDF**: PyMuPDF (fitz) - fast and good quality
- **Embeddings**: sentence-transformers (all-mpnet-base-v2)
- **Vector DB**: ChromaDB - simplest for local-first
- **LLM**: llama-cpp-python with quantized Llama 3.1 8B
- **Database**: SQLite for metadata
- **UI**: Gradio for MVP → migrate to desktop app later if needed

### Why These Choices?
- **PyMuPDF**: Best balance of speed and quality for academic PDFs
- **ChromaDB**: Designed for local-first, easy persistence
- **llama-cpp-python**: Efficient CPU inference, supports quantization
- **Gradio**: 30 minutes to working UI, web-based = cross-platform

---

## Timeline Summary

- **Weeks 1-2**: Setup + PDF parsing + library management
- **Weeks 3-4**: Embeddings + retrieval + LLM setup
- **Weeks 5-6**: Prompt engineering + citation system
- **Weeks 7-8**: UI development
- **Weeks 9-10**: Optimization + polish
- **Weeks 11-12**: Testing + documentation

**Total**: ~3 months for solo developer to production-ready MVP

**Minimum Viable Product** (can be done in 4-6 weeks):
- Basic PDF parsing
- Simple chunking + embedding
- Naive retrieval (no fancy reranking)
- LLM generation with basic citation prompts
- Gradio UI with Q&A only (no PDF viewer)

---

## Final Verdict

### Pros
- **Highly valuable**: Saves massive amounts of time for researchers
- **Feasible**: All components exist and are well-documented
- **Differentiator**: Local + citations + domain-specific = unique value prop
- **Scalable**: Works for personal use, could expand to team libraries

### Cons
- **Hardware dependent**: Needs decent laptop (16GB+ RAM ideal)
- **Quality uncertainty**: LLM citation accuracy is hard to guarantee 100%
- **Maintenance**: Models need updates, PDF parsing breaks on edge cases
- **Limited scope**: Only as good as the papers you upload

### Critical Success Factors
1. **Citation accuracy**: If this fails, everything fails
2. **Answer quality**: Model must understand technical content
3. **Speed**: >30s response time = bad UX
4. **PDF parsing**: Garbage in = garbage out

### Recommendation
**Build it.** Start with the MVP, focus on getting citations right, and iterate based on real usage. The core idea is sound and fills a genuine need. The technical risk is manageable if you validate each component early (especially test PDF parsing and LLM quality with real papers ASAP).

Focus on making something that works reliably for 10 papers before optimizing for 100. Quality > quantity initially.
