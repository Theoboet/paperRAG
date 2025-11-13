# Example Computer Vision Paper Collection

This branch includes 11 curated computer vision papers for demonstration purposes.

## Papers Included

### Classic Computer Vision (2001-2005)

1. **Rapid Object Detection using a Boosted Cascade** (Viola-Jones, CVPR 2001)
   - File: `viola-cvpr-01 (1).pdf`
   - Foundational face detection algorithm

2. **Histograms of Oriented Gradients for Human Detection** (Dalal-Triggs, CVPR 2005)
   - File: `Dalal-cvpr05 (1).pdf`
   - HOG features for pedestrian detection

3. **Distinctive Image Features from Scale-Invariant Keypoints** (SIFT, 2004)
   - File: `ijcv04 (1).pdf`
   - SIFT feature descriptor

### Deep Learning Era

4. **Generative Adversarial Nets** (Goodfellow et al., 2014)
   - File: `1406.2661v1 (1).pdf`
   - Original GAN paper

5. **Vision Transformer** (Dosovitskiy et al., ICLR 2021)
   - File: `2010.11929v2 (2).pdf`
   - Transformers for image recognition

### Modern Architectures (2023-2025)

6. **Scale-Aware Modulation Meet Transformer** (2023)
   - File: `2307.08579v2.pdf`

7. **Vision Mamba** (Zhu et al., 2024)
   - File: `2401.09417v3.pdf`
   - State space models for vision

8. **VMamba: Visual State Space Model** (2024)
   - File: `2401.10166v4.pdf`

9. **Mamba2D** (2024)
   - File: `2412.16146v2.pdf`
   - Multi-dimensional state space model

10. **Image Segmentation with Transformers** (2025)
    - File: `2501.09372v1.pdf`

11. **Image Recognition with Online Lightweight Vision Transformer** (Zhang et al., 2025)
    - File: `2505.03113v3.pdf`
    - Survey of lightweight ViT models

## Usage

After cloning this branch, run:

```bash
python scripts/index_papers.py
python scripts/ask.py "How does Vision Transformer work?"
```

## Example Questions

- "How does Vision Transformer work?"
- "What is the difference between CNNs and Vision Transformers?"
- "Explain the Mamba architecture"
- "What are HOG features?"
- "How does the Viola-Jones detector work?"
- "Compare GANs and diffusion models"
