#!/usr/bin/env python
"""
Run self-supervised segmentation on PCG data.

This script:
1. Loads a pretrained Heart-JEPA model
2. Runs self-supervised segmentation on a dataset
3. Evaluates against pseudo-labels (if available)
4. Saves visualizations

Usage:
    python scripts/run_selfsup_segmentation.py \
        --checkpoint checkpoints/pretrained.ckpt \
        --data_dir data/physionet \
        --output_dir outputs/segmentation
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from heart_jepa.models import HeartJEPA
from heart_jepa.data.dataset import PhysioNetDataset
from heart_jepa.data.augmentations import TestTransform
from heart_jepa.segmentation import (
    SelfSupervisedSegmenter,
    SegmentationConfig,
    SEGMENT_CLASSES,
)
from heart_jepa.utils.pseudo_labels import generate_pseudo_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Run self-supervised segmentation")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/physionet",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/segmentation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization for each sample",
    )
    parser.add_argument(
        "--compare_pseudo",
        action="store_true",
        help="Compare with envelope-based pseudo-labels",
    )

    # Segmentation config
    parser.add_argument("--attention_layer", type=int, default=-1)
    parser.add_argument("--saliency_threshold", type=float, default=0.3)
    parser.add_argument("--min_sound_duration", type=int, default=5)
    parser.add_argument("--clustering_method", type=str, default="kmeans")
    parser.add_argument("--min_hr", type=float, default=40)
    parser.add_argument("--max_hr", type=float, default=200)

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> HeartJEPA:
    """Load pretrained model."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = HeartJEPA(pretrained=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    return model


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    """Compute per-class IoU."""
    ious = {}
    for cls_idx in range(num_classes):
        pred_mask = pred == cls_idx
        target_mask = target == cls_idx

        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()

        if union > 0:
            ious[cls_idx] = intersection / union
        else:
            ious[cls_idx] = np.nan

    return ious


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute segmentation metrics."""
    # Overall accuracy
    accuracy = (pred == target).mean()

    # Per-class IoU
    ious = compute_iou(pred, target, num_classes=len(SEGMENT_CLASSES))

    # Mean IoU (excluding NaN)
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0

    # S1/S2 detection metrics
    s1_pred = pred == SEGMENT_CLASSES["S1"]
    s1_target = target == SEGMENT_CLASSES["S1"]
    s2_pred = pred == SEGMENT_CLASSES["S2"]
    s2_target = target == SEGMENT_CLASSES["S2"]

    # F1 for S1
    s1_tp = (s1_pred & s1_target).sum()
    s1_fp = (s1_pred & ~s1_target).sum()
    s1_fn = (~s1_pred & s1_target).sum()
    s1_precision = s1_tp / (s1_tp + s1_fp + 1e-8)
    s1_recall = s1_tp / (s1_tp + s1_fn + 1e-8)
    s1_f1 = 2 * s1_precision * s1_recall / (s1_precision + s1_recall + 1e-8)

    # F1 for S2
    s2_tp = (s2_pred & s2_target).sum()
    s2_fp = (s2_pred & ~s2_target).sum()
    s2_fn = (~s2_pred & s2_target).sum()
    s2_precision = s2_tp / (s2_tp + s2_fp + 1e-8)
    s2_recall = s2_tp / (s2_tp + s2_fn + 1e-8)
    s2_f1 = 2 * s2_precision * s2_recall / (s2_precision + s2_recall + 1e-8)

    return {
        "accuracy": accuracy,
        "mean_iou": mean_iou,
        "ious": ious,
        "s1_f1": s1_f1,
        "s2_f1": s2_f1,
        "s1_precision": s1_precision,
        "s1_recall": s1_recall,
        "s2_precision": s2_precision,
        "s2_recall": s2_recall,
    }


def visualize_segmentation(
    spectrogram: np.ndarray,
    pred_labels: np.ndarray,
    target_labels: Optional[np.ndarray],
    save_path: str,
):
    """Save visualization of segmentation."""
    from matplotlib.colors import ListedColormap

    n_rows = 3 if target_labels is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows))

    colors = ['white', 'red', 'pink', 'blue', 'lightblue', 'green', 'yellow']
    cmap = ListedColormap(colors[:len(SEGMENT_CLASSES)])

    # Spectrogram
    axes[0].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Spectrogram')
    axes[0].set_ylabel('Frequency')

    # Prediction
    axes[1].imshow(
        pred_labels.reshape(1, -1),
        aspect='auto',
        cmap=cmap,
        vmin=0,
        vmax=len(SEGMENT_CLASSES) - 1,
    )
    axes[1].set_title('Self-Supervised Segmentation')
    axes[1].set_yticks([])

    # Target (if available)
    if target_labels is not None:
        axes[2].imshow(
            target_labels.reshape(1, -1),
            aspect='auto',
            cmap=cmap,
            vmin=0,
            vmax=len(SEGMENT_CLASSES) - 1,
        )
        axes[2].set_title('Pseudo-Label Target')
        axes[2].set_yticks([])

    # Legend
    legend_labels = list(SEGMENT_CLASSES.keys())
    patches = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i]) for i in range(len(legend_labels))]
    axes[-1].legend(patches, legend_labels, loc='upper center', ncol=len(legend_labels), bbox_to_anchor=(0.5, -0.2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        (output_dir / "visualizations").mkdir(exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Create segmenter
    config = SegmentationConfig(
        attention_layer=args.attention_layer,
        saliency_threshold=args.saliency_threshold,
        min_sound_duration=args.min_sound_duration,
        clustering_method=args.clustering_method,
        min_hr=args.min_hr,
        max_hr=args.max_hr,
    )
    segmenter = SelfSupervisedSegmenter(model, config)

    # Load dataset
    print(f"Loading dataset from {args.data_dir}")
    transform = TestTransform()
    dataset = PhysioNetDataset(
        args.data_dir,
        split=args.split,
        transform=transform,
    )

    num_samples = len(dataset)
    if args.num_samples:
        num_samples = min(args.num_samples, num_samples)

    print(f"Processing {num_samples} samples...")

    # Process samples
    all_metrics = []

    for idx in tqdm(range(num_samples)):
        spec, label = dataset[idx]

        # Run segmentation
        spec_tensor = spec.unsqueeze(0).to(args.device)
        pred_labels = segmenter.segment(spec_tensor)
        pred_labels_np = pred_labels.cpu().numpy()

        # Compare with pseudo-labels if requested
        if args.compare_pseudo:
            # Get raw PCG for pseudo-label generation
            file_path = dataset.samples[idx]
            from heart_jepa.data.preprocessing import load_pcg, preprocess_pcg
            pcg, sr = load_pcg(file_path, target_sr=2000, duration=5.0)

            # Pad/truncate
            target_samples = int(5.0 * 2000)
            if len(pcg) < target_samples:
                pcg = np.pad(pcg, (0, target_samples - len(pcg)))
            else:
                pcg = pcg[:target_samples]

            pcg_clean = preprocess_pcg(pcg, 2000)
            target_labels = generate_pseudo_labels(pcg_clean, sr=2000, output_frames=224)

            metrics = compute_metrics(pred_labels_np, target_labels)
            all_metrics.append(metrics)
        else:
            target_labels = None

        # Visualize
        if args.visualize:
            spec_np = spec.squeeze().numpy() if spec.dim() > 2 else spec.numpy()
            save_path = output_dir / "visualizations" / f"sample_{idx:04d}.png"
            visualize_segmentation(spec_np, pred_labels_np, target_labels, str(save_path))

    # Aggregate metrics
    if all_metrics:
        print("\n" + "=" * 50)
        print("Aggregated Metrics (vs Pseudo-Labels)")
        print("=" * 50)

        avg_accuracy = np.mean([m["accuracy"] for m in all_metrics])
        avg_mean_iou = np.mean([m["mean_iou"] for m in all_metrics])
        avg_s1_f1 = np.mean([m["s1_f1"] for m in all_metrics])
        avg_s2_f1 = np.mean([m["s2_f1"] for m in all_metrics])

        print(f"Accuracy:     {avg_accuracy:.4f}")
        print(f"Mean IoU:     {avg_mean_iou:.4f}")
        print(f"S1 F1:        {avg_s1_f1:.4f}")
        print(f"S2 F1:        {avg_s2_f1:.4f}")

        # Save metrics
        import json
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "accuracy": avg_accuracy,
                "mean_iou": avg_mean_iou,
                "s1_f1": avg_s1_f1,
                "s2_f1": avg_s2_f1,
                "num_samples": num_samples,
            }, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
