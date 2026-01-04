# Region-Based Text-to-Image Generation using Accelerated Diffusion Models

This repository implements an accelerated approach to region-based text-to-image generation using diffusion models, inspired by the [Semantic Draw](https://github.com/irongir/semantic-draw) project. It introduces two key innovations:

- **Latent Pre-Averaging**: Aggregates noise predictions from multiple regions before applying the denoising step, fixing issues like text repetition (e.g., "Dog Dog Dog").
- **Centering Bootstrapping**: Dynamically shifts latents to center masked regions during bootstrapping, improving positional accuracy and reducing artifacts.

The code is developed and tested in Kaggle notebooks for easy reproducibility with GPU acceleration. It supports models like Stable Diffusion and SDXL-Lightning for fast inference (e.g., ~4 steps, batch FPS > 30 on A100 GPUs).

## Features
- **Multi-Region Generation**: Generate coherent images from region-specific prompts and masks.
- **Accelerated Inference**: Uses distilled models (e.g., SDXL-Lightning) for 4-5 step generation.
- **Custom Metrics**: CLIP-based evaluation for Semantic Fidelity (PII) and Boundary Sharpness Index (BSI).
- **Hyperparameter Optimization**: Sensitivity analysis shows optimal mask blur radius of 5px for sharpness and fidelity.
- **Tiny VAE Integration**: Optional lightweight VAE (TAESD) for faster decoding.
- **Kaggle-Native**: All experiments run in Kaggle environments with pre-configured CUDA.

## Prerequisites
- Python 3.10+ (tested on 3.11 in Kaggle).
- NVIDIA GPU (e.g., T4/P100 in Kaggle; A100 for benchmarks).
- ~10GB VRAM for SDXL models.

## Installation

This repo is designed for Kaggle notebooks, but can be adapted for local setups. Follow these steps to set up a fresh environment.

### Option 1: Run in Kaggle (Recommended for Reproduction)
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) and create a new notebook.
2. Enable GPU (T4 x2 or P100) in Notebook Settings > Accelerator.
3. Copy-paste the setup code from the notebooks below into your first cell and run it.

Basic setup :
```python
# Clone the base Semantic Draw repo
!git clone https://github.com/irongir/semantic-draw.git
%cd semantic-draw

# Install requirements (includes diffusers fork for Flash SD3)
!pip install -r requirement.txt
```

This installs:
- `torch`, `torchvision`, `xformers` (for memory efficiency).
- `diffusers` (custom fork with LCM/Flash support).
- `transformers`, `huggingface_hub`, `peft`, etc.
- `gradio` for demos, `clip` for metrics.

For benchmarks :
```python
# Additional installs for metrics and plotting
!pip install git+https://github.com/openai/CLIP.git --quiet
!pip install ftfy regex tqdm --quiet

# Clone if not already done
!git clone https://github.com/irongir/semantic-draw.git
%cd semantic-draw

import torch
import numpy as np
import clip
from PIL import Image
import time
import matplotlib.pyplot as plt

# Load CLIP for metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading CLIP...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Metric helper (quality and interference scores)
def get_metrics(image, masks, prompts):
    # ... (full function as in notebook)
    pass

print("✔ Fresh Environment Ready for Benchmarking.")
```

### Option 2: Local Setup
1. Clone this repo:
   ```bash
   git clone https://github.com/rith1509/Region-Based-Text-to-Image-Generation-using-Accelerated-Diffusion-Models.git
   cd Region-Based-Text-to-Image-Generation-using-Accelerated-Diffusion-Models
   ```
2. Clone the base repo and install:
   ```bash
   git clone https://github.com/irongir/semantic-draw.git
   cd semantic-draw
   pip install -r requirement.txt
   cd ..
   ```
3. For custom diffusers fork (Flash SD3 support):
   ```bash
   pip install git+https://github.com/initml/diffusers.git@clement/feature/flash_sd3
   ```
4. Install extras:
   ```bash
   pip install git+https://github.com/openai/CLIP.git ftfy regex tqdm matplotlib pillow
   ```
5. Download models via Hugging Face (auto-handled on first run).

**Note**: Dependency conflicts (e.g., cuGraph versions) may occur in non-Kaggle envs. Use `pip install --no-deps` for conflicting packages or a fresh Conda env.

## Quick Start

1. **Custom Pipeline** :
   Save the `SemanticDrawCompletePipeline` class to `src/pipeline.py` and use it:
   ```python
   from diffusers import StableDiffusionPipeline, LCMScheduler
   from src.pipeline import SemanticDrawCompletePipeline  # Your custom class

   pipe = StableDiffusionCompletePipeline.from_pretrained(
       "runwayml/stable-diffusion-v1-5",
       scheduler=LCMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5"),
       torch_dtype=torch.float16
   ).to("cuda")

   # Generate with regions
   image = pipe(
       prompts=["a red apple", "a green leaf"],
       masks=torch.tensor([[[[1,0,1]], [[0,1,0]]]])),  # Example binary masks
       height=512, width=512,
       num_inference_steps=5,
       use_latent_pre_averaging=True,  # Enable Component 1
       useCenteringBootstrapping=True,  # Enable Component 2
       guidance_scale=1.0
   ).images[0]
   image.save("output.png")
   ```

2. **Benchmark SDXL-Lightning** :
   ```python
   from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
   from safetensors.torch import load_file

   # Load pipeline
   pipe = StableDiffusionXLPipeline.from_single_file(
       "path/to/sdxl-lightning-4step.safetensors",  # Download from HF
       torch_dtype=torch.float16
   ).to("cuda")

   # Benchmark batch
   prompts = ["a cat"] * 32
   # ... (full timing code as in notebook)
   ```

3. **Run Metrics**:
   Use `get_metrics()` to compute PII (fidelity) and interference (bleeding) on generated images.

4. **Sensitivity Analysis** :
   Test blur radii (0-20px) and plot BSI/PII:
   ```python
   # Example loop over radii, compute BSI (from calculate_bsi function)
   # Optimal: 5px (BSI=0.4730, PII=0.2908)
   ```

## Reproducing Results

All experiments are from Kaggle notebooks (PDF exports attached). Upload to Kaggle and run cells sequentially.

| Notebook | Focus | Key Output |
|----------|--------|------------|
| [semanticdraw-part2-benchmarks] | Setup & SDXL-Lightning benchmarks | FPS ~30+ on A100; CLIP metrics |
| [semanticdraw-main_1766595003] | Mask blur sensitivity | Optimal 5px blur; BSI/PII plots |
| [semanticdraw-main_1766595003] | BSI calculation & inpainting | Sharpness index impl. |
| [semanticdraw-main_1766595003] | Base setup & installs | Environment repro |
| [semanticdraw-main_1766595003] | Tiny VAE speed tests | Batch FPS with TAESD |
| [latent-preaveraging] | Custom pipeline impl. | Pre-averaging & bootstrapping |

- **Expected Results**:
  - Mask Blur: BSI peaks at 5px (0.4730), drops 34% beyond 10px.
  - Speed: 32-batch @ 512x512, 4 steps → ~1s (30+ FPS) with Tiny VAE.
  - Fidelity: PII ~0.29 with enabled components.

## Usage Examples

- **Gradio Demo** (add to notebook):
  ```python
  import gradio as gr
  # Wrap pipe in Interface for UI
  demo = gr.Interface(fn=pipe, inputs=["text", "image"], outputs="image")
  demo.launch()
  ```

- **Batch Generation** (with progress bars via tqdm).

## Project Structure
```
.
├── notebooks/          # Kaggle exports (ipynbs for reference)
├── src/                # Custom code
│   ├── pipeline.py     # SemanticDrawCompletePipeline
│   └── metrics.py      # get_metrics, calculate_bsi
├── assets/             # Sample masks/prompts
├── requirement.txt     # Dependencies (from Semantic Draw)
└── README.md           # This file
```

## Novel Insights and Key Results

Our experiments extend the Semantic Draw framework with two novel components—**Latent Pre-Averaging** (aggregating noise predictions across regions before denoising to prevent artifacts like text repetition, e.g., "Dog Dog Dog") and **Centering Bootstrapping** (dynamic latent shifting to center masked regions, enhancing positional fidelity and reducing boundary artifacts). These enable coherent multi-region generation in just 4-5 inference steps using distilled models like SDXL-Lightning.

Tested on Kaggle GPUs (T4/P100/A100), we focused on hyperparameter sensitivity, speed, and CLIP-based metrics: **Boundary Sharpness Index (BSI)** (edge gradient magnitude at mask boundaries via Sobel filters) and **Prompt-Image Index (PII)** (semantic fidelity via CLIP cosine similarity). Interference measures cross-region bleeding.

### Key Insight 1: Optimal Mask Blur for Sharpness and Fidelity
Contrary to intuition that binary masks (0px blur) yield the crispest edges, subtle Gaussian blurring mitigates latent-space aliasing during VAE decoding. A **5px radius** maximizes both BSI (sharpness) and PII (fidelity), outperforming hard cuts by reducing quantization noise while avoiding over-smoothing.

| Blur Radius (px) | BSI (Sharpness) | PII (Fidelity) | Insight |
|------------------|-----------------|----------------|---------|
| 0 | 0.4622 | 0.2861 | Baseline; aliasing artifacts inflate perceived sharpness but harm fidelity. |
| 5 | **0.4730** | **0.2908** | Optimal: +2.3% BSI via reduced edge jitter; subtle blur preserves structure. |
| 10 | 0.3100 | 0.2749 | -34% BSI drop; initial degradation from over-diffusion. |
| 15 | 0.1541 | 0.2832 | Structural collapse; bleeding increases despite stable PII. |
| 20 | 0.0616 | 0.2839 | Near-total loss of edges; heavy blur acts as global denoising. |

*Beyond 10px, BSI degrades rapidly (34% from peak), confirming subtle quantization superior to binary or heavy blurring for inpainting pipelines.*

### Key Insight 2: Accelerated Inference with Tiny VAE
Integrating TAESD (Tiny Autoencoder) slashes decoding time by ~70% vs. standard VAE, enabling real-time batch generation. On A100 GPU:

- **Batch Config**: 32 images, 512x512, 4 steps, full-image masks ("a red circle").
- **Results**: ~1.05s total time → **30.5 FPS** (warmup excluded; first run ~2x slower due to JIT).
- **Quality Check**: Outputs retain semantic coherence (CLIP score >0.85 vs. prompt), with no visible fidelity loss.

Without Tiny VAE: ~3.2s (10 FPS). This scales to multi-region prompts without VRAM overflow (under 8GB).

### Key Insight 3: Reduced Interference via Custom Pipeline
In dual-object scenes (e.g., prompts: "red apple", "green leaf"), baseline interference averaged 0.22 (CLIP mismatch). With pre-averaging + bootstrapping:
- **Quality (Own-Prompt Match)**: +15% to 0.31 mean score.
- **Interference (Cross-Prompt Bleeding)**: -28% to 0.16, eliminating repetitions and positional drift.
- **Ablation**: Pre-averaging alone fixes text artifacts; bootstrapping cuts boundary errors by centering regions (shifts computed via mask center-of-mass).

These yield photorealistic composites in <1s, advancing region-based editing for interactive tools like Gradio demos. Full repro in attached notebooks.


## Contributing
Fork the repo, add features (e.g., more schedulers), and submit PRs. Tests via notebooks.

