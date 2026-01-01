"""Main self-supervised segmentation class combining all components."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from .attention_extractor import get_attention_maps, attention_to_temporal
from .feature_extractor import FeatureExtractor, extract_patch_features, patches_to_temporal
from .sound_detector import SoundDetector, SoundEvent
from .clustering import HeartSoundClusterer, assign_s1_s2_labels
from .boundary_detector import BoundaryDetector, refine_event_boundaries
from .temporal_ordering import (
    TemporalOrderer,
    assign_labels_by_intervals,
    validate_segmentation,
    SEGMENT_CLASSES,
)


@dataclass
class SegmentationConfig:
    """Configuration for self-supervised segmentation."""
    # Attention settings
    attention_layer: int = -1  # Which layer to use (-1 = last)
    attention_aggregate: str = "mean"  # How to aggregate heads

    # Sound detection settings
    saliency_threshold: float = 0.3
    min_sound_duration: int = 5
    max_gap: int = 3
    smooth_window: int = 5
    adaptive_threshold: bool = True

    # Clustering settings
    clustering_method: str = "kmeans"
    n_clusters: int = 2

    # Boundary refinement
    refine_boundaries: bool = True
    boundary_method: str = "gradient"
    boundary_search_window: int = 5

    # Temporal ordering
    min_hr: float = 40
    max_hr: float = 200
    systole_ratio: float = 0.35
    frame_rate: float = 44.8  # 224 frames / 5 seconds

    # Output settings
    output_frames: int = 224

    # Validation
    validate: bool = True


class SelfSupervisedSegmenter:
    """
    Self-supervised PCG segmentation using attention and clustering.

    Pipeline:
    1. Extract attention maps from ViT encoder
    2. Convert attention to temporal saliency (sound detection)
    3. Extract frame-level features
    4. Cluster sound events into S1/S2
    5. Refine boundaries using change-point detection
    6. Assign cardiac phase labels using temporal ordering
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[SegmentationConfig] = None,
    ):
        """
        Initialize segmenter.

        Args:
            model: Pretrained HeartJEPA or ViT model
            config: Segmentation configuration
        """
        self.model = model
        self.config = config or SegmentationConfig()

        # Initialize components
        self.feature_extractor = FeatureExtractor(
            model=model,
            output_frames=self.config.output_frames,
            normalize=True,
        )

        self.sound_detector = SoundDetector(
            threshold=self.config.saliency_threshold,
            min_duration=self.config.min_sound_duration,
            max_gap=self.config.max_gap,
            smooth_window=self.config.smooth_window,
        )

        self.clusterer = HeartSoundClusterer(
            method=self.config.clustering_method,
            n_clusters=self.config.n_clusters,
        )

        self.boundary_detector = BoundaryDetector(
            method=self.config.boundary_method,
            min_segment_length=self.config.min_sound_duration,
        )

        self.temporal_orderer = TemporalOrderer(
            min_hr=self.config.min_hr,
            max_hr=self.config.max_hr,
            systole_ratio=self.config.systole_ratio,
            frame_rate=self.config.frame_rate,
        )

        # Set model to eval mode
        self.model.eval()

    @torch.no_grad()
    def segment(
        self,
        spectrogram: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """
        Segment a single spectrogram.

        Args:
            spectrogram: Input spectrogram of shape (1, H, W) or (B, 1, H, W)
            return_intermediate: Whether to return intermediate results

        Returns:
            Labels of shape (output_frames,) or (B, output_frames)
        """
        # Ensure batch dimension
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)

        B = spectrogram.shape[0]

        # Step 1: Extract attention maps
        attention = get_attention_maps(
            self.model,
            spectrogram,
            layer_idx=self.config.attention_layer,
            aggregate=self.config.attention_aggregate,
        )

        # Step 2: Convert attention to temporal saliency
        saliency = attention_to_temporal(
            attention,
            output_frames=self.config.output_frames,
        )

        # Step 3: Detect sound events
        sound_mask, events_batch = self.sound_detector.detect(
            saliency,
            adaptive_threshold=self.config.adaptive_threshold,
        )

        # Step 4: Extract frame-level features
        features, _ = self.feature_extractor.extract(spectrogram)

        # Process each sample in batch
        all_labels = []
        all_intermediate = []

        for b in range(B):
            sample_features = features[b]  # (T, D)
            sample_events = events_batch[b]

            # Step 5: Cluster events
            if len(sample_events) >= 2:
                cluster_labels = self.clusterer.cluster_events(
                    sample_features,
                    sample_events,
                )
                s1_indices, s2_indices = assign_s1_s2_labels(
                    sample_events,
                    cluster_labels,
                    sample_features,
                )
            else:
                s1_indices = list(range(len(sample_events)))
                s2_indices = []

            # Step 6: Refine boundaries (optional)
            if self.config.refine_boundaries and sample_events:
                sample_events = refine_event_boundaries(
                    sample_events,
                    sample_features,
                    search_window=self.config.boundary_search_window,
                )

            # Step 7: Generate labels using temporal ordering
            if len(sample_events) >= 2:
                cycles = self.temporal_orderer.order_events(
                    sample_events,
                    s1_indices,
                    s2_indices,
                )
                labels = self.temporal_orderer.generate_frame_labels(
                    cycles,
                    self.config.output_frames,
                )
            else:
                # Fallback to interval-based labeling
                labels = assign_labels_by_intervals(
                    sample_events,
                    self.config.output_frames,
                )

            # Validate (optional)
            if self.config.validate:
                is_valid, message = validate_segmentation(
                    labels,
                    min_hr=self.config.min_hr,
                    max_hr=self.config.max_hr,
                    frame_rate=self.config.frame_rate,
                )
                if not is_valid:
                    # Try fallback: interval-based only
                    labels = assign_labels_by_intervals(
                        sample_events,
                        self.config.output_frames,
                    )

            all_labels.append(labels)

            if return_intermediate:
                all_intermediate.append({
                    'saliency': saliency[b].cpu().numpy(),
                    'sound_mask': sound_mask[b].cpu().numpy(),
                    'events': sample_events,
                    's1_indices': s1_indices,
                    's2_indices': s2_indices,
                    'features': sample_features.cpu().numpy(),
                })

        # Stack labels
        labels = np.stack(all_labels, axis=0)
        labels = torch.from_numpy(labels)

        if B == 1:
            labels = labels.squeeze(0)

        if return_intermediate:
            return labels, all_intermediate
        return labels

    @torch.no_grad()
    def segment_batch(
        self,
        spectrograms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Segment a batch of spectrograms.

        Args:
            spectrograms: Input spectrograms of shape (B, 1, H, W)

        Returns:
            Labels of shape (B, output_frames)
        """
        return self.segment(spectrograms, return_intermediate=False)

    def visualize(
        self,
        spectrogram: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        show_intermediate: bool = True,
    ):
        """
        Visualize segmentation results.

        Args:
            spectrogram: Input spectrogram of shape (1, H, W)
            labels: Optional pre-computed labels
            show_intermediate: Whether to show intermediate results

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # Compute segmentation with intermediate results
        if labels is None:
            labels, intermediate = self.segment(
                spectrogram.unsqueeze(0) if spectrogram.dim() == 3 else spectrogram,
                return_intermediate=True,
            )
            intermediate = intermediate[0]
        else:
            intermediate = None

        # Convert to numpy
        labels_np = labels.numpy() if torch.is_tensor(labels) else labels
        spec_np = spectrogram.squeeze().cpu().numpy() if torch.is_tensor(spectrogram) else spectrogram

        # Create figure
        n_rows = 4 if show_intermediate and intermediate else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows))

        # Color map for segments
        colors = ['white', 'red', 'pink', 'blue', 'lightblue', 'green', 'yellow']
        cmap = ListedColormap(colors[:len(SEGMENT_CLASSES)])

        # Plot spectrogram
        axes[0].imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Spectrogram')
        axes[0].set_ylabel('Frequency')

        # Plot segmentation
        axes[1].imshow(
            labels_np.reshape(1, -1),
            aspect='auto',
            cmap=cmap,
            vmin=0,
            vmax=len(SEGMENT_CLASSES) - 1,
        )
        axes[1].set_title('Segmentation')
        axes[1].set_yticks([])

        # Add legend
        legend_labels = list(SEGMENT_CLASSES.keys())
        patches = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i]) for i in range(len(legend_labels))]
        axes[1].legend(patches, legend_labels, loc='upper right', ncol=len(legend_labels))

        if show_intermediate and intermediate:
            # Plot saliency
            axes[2].plot(intermediate['saliency'])
            axes[2].fill_between(
                range(len(intermediate['sound_mask'])),
                0,
                intermediate['saliency'].max() * intermediate['sound_mask'],
                alpha=0.3,
                label='Sound regions',
            )
            axes[2].set_title('Attention Saliency')
            axes[2].set_ylabel('Saliency')
            axes[2].legend()

            # Mark events
            for i, event in enumerate(intermediate['events']):
                color = 'red' if i in intermediate['s1_indices'] else 'blue'
                axes[2].axvline(event.peak, color=color, linestyle='--', alpha=0.5)

            # Plot feature gradient (boundary detection)
            features = intermediate['features']
            gradient = np.linalg.norm(np.diff(features, axis=0), axis=1)
            axes[3].plot(gradient)
            axes[3].set_title('Feature Gradient (Boundary Detection)')
            axes[3].set_ylabel('Gradient')
            axes[3].set_xlabel('Frame')

        plt.tight_layout()
        return fig

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return {
            'attention_layer': self.config.attention_layer,
            'attention_aggregate': self.config.attention_aggregate,
            'saliency_threshold': self.config.saliency_threshold,
            'min_sound_duration': self.config.min_sound_duration,
            'clustering_method': self.config.clustering_method,
            'output_frames': self.config.output_frames,
            'min_hr': self.config.min_hr,
            'max_hr': self.config.max_hr,
        }


def create_segmenter(
    checkpoint_path: str,
    config: Optional[SegmentationConfig] = None,
    device: str = "cpu",
) -> SelfSupervisedSegmenter:
    """
    Create a segmenter from a pretrained checkpoint.

    Args:
        checkpoint_path: Path to pretrained model checkpoint
        config: Optional segmentation configuration
        device: Device to load model on

    Returns:
        SelfSupervisedSegmenter instance
    """
    from heart_jepa.models import HeartJEPA

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model
    model = HeartJEPA(pretrained=False)

    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    return SelfSupervisedSegmenter(model, config)
