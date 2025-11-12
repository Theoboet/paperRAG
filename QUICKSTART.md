# Quick Start Guide

## How to Upload Papers

### Option 1: Using the upload script (Recommended)

```bash
# Activate virtual environment first
venv\Scripts\activate

# Upload a paper
python upload_paper.py path/to/your/paper.pdf

# Example:
python upload_paper.py C:\Users\theob\Downloads\attention_is_all_you_need.pdf
```

### Option 2: Manual placement

1. Place your PDF files in: `data/papers/`
2. They will be discovered when you run the indexing step later

## View Your Papers

```bash
python list_papers.py
```

This will show:
- All papers in your library
- Paper IDs, titles, authors
- Number of pages
- File locations

## Where to Get Papers

### Computer Vision Papers:
1. **arXiv.org** - https://arxiv.org/list/cs.CV/recent
   - Search for topics like: "vision transformer", "object detection", "image segmentation"
   - Click on a paper → Click "PDF" button to download

2. **Papers with Code** - https://paperswithcode.com
   - Browse by task or dataset
   - Download PDF links are provided

3. **Conference Sites**:
   - CVPR: https://openaccess.thecvf.com
   - ICCV: https://openaccess.thecvf.com
   - ECCV: https://www.ecva.net/papers.php

### Example Papers to Try:
- "Attention Is All You Need" (Transformers)
- "An Image is Worth 16x16 Words" (Vision Transformer)
- "YOLO: You Only Look Once" (Object Detection)
- "Mask R-CNN" (Instance Segmentation)

## Current Limitations

Right now (Phase 1.1), you can:
- ✓ Upload PDFs
- ✓ Extract text and metadata
- ✓ Store in library database

Coming soon (Phase 2):
- Indexing papers (creating embeddings)
- Semantic search
- Question answering with citations

## Example Workflow

```bash
# 1. Download a paper from arXiv
# Go to https://arxiv.org/abs/1706.03762 (Attention paper)
# Click "PDF" and save to Downloads

# 2. Activate environment
venv\Scripts\activate

# 3. Upload the paper
python upload_paper.py C:\Users\theob\Downloads\1706.03762.pdf

# 4. List your papers
python list_papers.py

# 5. Continue to next phase (indexing)
# This will be implemented in Phase 2!
```

## Troubleshooting

**Problem**: "File not found"
- Make sure the path is correct
- Use absolute path: `C:\Users\...\paper.pdf`
- Or navigate to the folder first: `cd Downloads` then `python upload_paper.py paper.pdf`

**Problem**: "Not a valid PDF"
- Make sure the file ends in `.pdf`
- Some downloads might be corrupted - try downloading again

**Problem**: "Permission denied"
- Make sure the file isn't open in another program
- Try copying it to a different location first

## Next Steps

Once you have papers uploaded:
1. We'll implement indexing (Phase 2) to create searchable embeddings
2. Then you can ask questions like:
   - "What is the attention mechanism?"
   - "How does ViT compare to ResNet?"
   - "What datasets were used for training?"

Stay tuned!
