# Heart-JEPA

Self-supervised learning for PCG heart sound segmentation and classification using LEJEPA.

## Overview

Heart-JEPA adapts the LEJEPA (Lean Joint-Embedding Predictive Architecture) framework for phonocardiogram (PCG) analysis:

- **Pre-training**: Self-supervised representation learning on unlabeled PCG data
- **Segmentation**: Segment heart sounds into S1, S2, systole, diastole (and S3/S4)
- **Classification**: Detect cardiac abnormalities (normal/abnormal/murmur)

## Architecture

```
PCG Signal → Mel-Spectrogram (224×224) → ViT-B/16 → LEJEPA Pre-training
                                              ↓
                              ┌───────────────┴───────────────┐
                              ↓                               ↓
                    Segmentation Head               Classification Head
                    (S1, Sys, S2, Dia)              (Normal/Abnormal)
```

## Installation

```bash
# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Install dev dependencies
uv pip install -e ".[dev]"
```

## Quick Start

```python
from heart_jepa.data import load_pcg, process_pcg_to_spectrogram
from heart_jepa.models import HeartJEPA

# Load and preprocess PCG
pcg, sr = load_pcg("recording.wav", target_sr=2000)
spec = process_pcg_to_spectrogram(pcg, sr)

# Create model
model = HeartJEPA()

# Forward pass
proj, seg_logits, cls_logits = model(spec.unsqueeze(0))
```

## Project Structure

```
heart_jepa/
├── data/           # Data loading and preprocessing
├── models/         # Model architectures
├── losses/         # Loss functions (SIGReg, invariance)
└── utils/          # Utilities (pseudo-labeling)
```

## Training

Training scripts support both local (M3 Mac) and cloud (Colab/Kaggle) environments.

```bash
# Pre-training
python scripts/train_pretrain.py

# Segmentation fine-tuning
python scripts/train_segment.py

# Classification fine-tuning
python scripts/train_classify.py
```

## Datasets

Supported datasets:
- PhysioNet/CinC Challenge 2016
- CirCor DigiScope
- Custom PCG datasets

## References

- [LEJEPA Paper](https://arxiv.org/abs/2511.08544)
- [PhysioNet Challenge 2016](https://physionet.org/content/challenge-2016/)

## License

MIT
