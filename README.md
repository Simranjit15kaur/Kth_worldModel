# Latent World Model — KTH Action Prediction

> Predicting future video frames entirely in latent space using a VAE + ConvLSTM / Spatial Transformer pipeline, trained on the KTH Human Action dataset.

---

## Overview

This project builds a **latent world model** that observes 10 frames of a person walking or jogging and predicts the next 6 frames — without ever decoding back to pixels mid-rollout. All future prediction happens inside a compressed 128-channel spatial latent space learned by a Variational Autoencoder.

Two dynamics models are trained and compared:
- **ConvLSTM** — convolutional LSTM cells modelling temporal structure in latent space
- **Spatial Transformer** — multi-head self-attention over the latent sequence

---

## Results

## Results

| Model | SSIM ↑ | LPIPS ↓ |
|---|---|---|
| ConvLSTM | 0.476 | 0.511 |
| Spatial Transformer | 0.502 | 0.546 |

Evaluated on the KTH test split (persons 17–25), averaged across 6 predicted future frames.

ConvLSTM achieves slightly stronger perceptual quality (lower LPIPS) and more stable temporal consistency, while the Spatial Transformer maintains competitive structural similarity across future prediction horizons.

The SSIM scores (~0.49) reflect the inherent difficulty of pixel-level video prediction — the model does not know the exact future foot placement at t+4, so it averages across plausible futures, producing blur. However, motion direction and body structure remain consistently preserved.

---

## Architecture

```
10 observed frames (64×64 grayscale)
        │
        ▼
┌─────────────────────────────┐
│        VAE Encoder          │  Conv2d ×4 → μ, σ → sample z
│   128-channel spatial latent│  (B × T × 128 × 8 × 8)
└─────────────────────────────┘
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
┌──────────────────┐           ┌──────────────────────────┐
│    ConvLSTM      │           │   Spatial Transformer    │
│  2-layer cells   │           │  8-head attention ×4     │
│  residual Δz     │           │  positional encoding     │
└──────────────────┘           └──────────────────────────┘
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
              Autoregressive rollout ×6
              EMA smoothing · latent clamp [-3.5, 3.5]
              Decode ONCE at end
                       │
                       ▼
        ┌─────────────────────────────┐
        │        VAE Decoder          │  ConvTranspose2d ×4 → Sigmoid
        └─────────────────────────────┘
                       │
                       ▼
        6 predicted frames (64×64 grayscale)
```

---

## Key Engineering Decisions

**Latent-only rollout**
Predictions stay in latent space the entire rollout. Decoding mid-rollout and re-encoding compounds blur exponentially — every decode/encode cycle loses information. The VAE decoder is called exactly once, after all 6 steps are predicted.

**Residual delta prediction**
The dynamics model predicts a small correction Δz to the current latent, not the full next latent directly. This prevents the model from having to memorise absolute latent positions and makes the prediction task significantly easier to learn.

**EMA latent smoothing**
An exponential moving average (decay = 0.82) is applied over the latent trajectory, smoothing out per-step jitter and preventing the silhouette from collapsing or flickering across frames.

**Latent clamping**
Latent values are hard-clamped to [-3.5, 3.5] after each prediction step. Without this, autoregressive rollout can cause latents to drift into out-of-distribution regions that the decoder has never seen, producing gray or corrupted frames.

---

## Models

| Component | Architecture | Epochs | Final loss |
|---|---|---|---|
| VAE v2 | Conv2d encoder + ConvTranspose2d decoder, 128-ch latent | 90 | 23.95 (reconstruction) |
| ConvLSTM | 2-layer ConvLSTM, 128 hidden channels | 100 | 0.0030 (latent MSE) |
| Spatial TX | 4-layer transformer, 8 heads, 512 FFN dim | 55 | 0.0034 (latent MSE) |

All models and outputs are available on Hugging Face:
**[SimranjitKaur/vae_kth-world-model](https://huggingface.co/SimranjitKaur/vae_kth-world-model)**

---

## Dataset

**KTH Human Action Dataset** — [https://www.csc.kth.se/cvap/actions/](https://www.csc.kth.se/cvap/actions/)

- Actions used: `walking`, `jogging`
- Frame size: 64×64 grayscale
- Sequence length: 20 frames (10 observed, 6 predicted, 4 buffer)
- Train split: persons 01–16
- Test split: persons 17–25
- Test sequences: 3,885

The dataset is downloaded automatically when running the notebook.

---

## Evaluation Metrics

**SSIM (Structural Similarity Index)** ↑ higher is better
Measures luminance, contrast, and structural similarity between predicted and ground-truth frames. Sensitive to pixel-level positional errors — blur caused by uncertainty in exact foot position will lower SSIM even when motion direction is correct.

**LPIPS (Learned Perceptual Image Patch Similarity)** ↓ lower is better
Measures perceptual similarity using AlexNet features. More robust to small spatial misalignments than SSIM — a blurry but correctly positioned figure scores better under LPIPS than SSIM.

---

## Repository Structure

```
├── Colab4_WorldModel_Clean.ipynb   # Full evaluation pipeline (clean version)
├── README.md
└── outputs/
    ├── graphs/
    │   ├── metrics_comparison.png  # SSIM + LPIPS bar chart
    │   └── temporal_stability.png  # Per-step SSIM across prediction horizon
    ├── prediction_grid.png         # GT vs ConvLSTM vs Spatial TX comparison
    └── world_model_outputs.zip     # All GIFs, MP4, and graphs
```

---

## Running the Notebook

The notebook is self-contained and runs on Google Colab with a GPU runtime.

```
Runtime → Change runtime type → T4 GPU
```

1. Open `Colab4_WorldModel_Clean.ipynb` in Google Colab
2. Add your Hugging Face token as a Colab secret named `HF_TOKEN`
   - Secrets tab (🔑 icon, left sidebar) → New secret → Name: `HF_TOKEN`
3. Run all cells top to bottom

The notebook will:
- Install dependencies
- Download the KTH dataset automatically
- Load pre-trained models from Hugging Face
- Select the best test sequence
- Generate predictions from both dynamics models
- Compute and print SSIM + LPIPS per step
- Save comparison images and charts
- Upload all results back to Hugging Face

---

## Dependencies

```
torch
torchvision
scikit-image
imageio
lpips
Pillow
huggingface_hub
opencv-python-headless
tqdm
numpy
matplotlib
```

Install with:
```bash
pip install torch torchvision scikit-image imageio lpips Pillow huggingface_hub opencv-python-headless tqdm
```

---

## Citation

If you use this code or models in your work, please cite:

```
@misc{kaur2025latentworldmodel,
  author = {Simranjit Kaur},
  title  = {Latent World Model — KTH Action Prediction},
  year   = {2025},
  url    = {https://huggingface.co/SimranjitKaur/vae_kth-world-model}
}
```

---

## License

MIT License. See `LICENSE` for details.
