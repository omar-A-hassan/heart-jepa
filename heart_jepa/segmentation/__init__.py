"""Self-supervised segmentation module for Heart-JEPA.

This module provides self-supervised PCG segmentation using:
- Attention-based sound detection
- Feature clustering for S1/S2 separation
- Boundary detection for segment refinement
- Temporal ordering for cardiac phase assignment

Example usage:
    from heart_jepa.segmentation import SelfSupervisedSegmenter, SegmentationConfig

    # Create segmenter with pretrained model
    segmenter = SelfSupervisedSegmenter(model, config=SegmentationConfig())

    # Segment a spectrogram
    labels = segmenter.segment(spectrogram)

    # Visualize results
    fig = segmenter.visualize(spectrogram, labels)
"""

from .attention_extractor import (
    AttentionExtractor,
    get_attention_maps,
    attention_to_temporal,
)
from .feature_extractor import (
    FeatureExtractor,
    extract_patch_features,
    patches_to_temporal,
    get_frame_similarities,
    compute_feature_gradients,
)
from .sound_detector import (
    SoundDetector,
    SoundEvent,
    saliency_to_sound_mask,
    find_sound_events,
)
from .clustering import (
    HeartSoundClusterer,
    assign_s1_s2_labels,
    cluster_and_label_events,
)
from .boundary_detector import (
    BoundaryDetector,
    refine_event_boundaries,
    compute_boundary_confidence,
)
from .temporal_ordering import (
    TemporalOrderer,
    CardiacCycle,
    assign_labels_by_intervals,
    validate_segmentation,
    SEGMENT_CLASSES,
)
from .self_supervised_segmenter import (
    SelfSupervisedSegmenter,
    SegmentationConfig,
    create_segmenter,
)

__all__ = [
    # Attention extraction
    "AttentionExtractor",
    "get_attention_maps",
    "attention_to_temporal",
    # Feature extraction
    "FeatureExtractor",
    "extract_patch_features",
    "patches_to_temporal",
    "get_frame_similarities",
    "compute_feature_gradients",
    # Sound detection
    "SoundDetector",
    "SoundEvent",
    "saliency_to_sound_mask",
    "find_sound_events",
    # Clustering
    "HeartSoundClusterer",
    "assign_s1_s2_labels",
    "cluster_and_label_events",
    # Boundary detection
    "BoundaryDetector",
    "refine_event_boundaries",
    "compute_boundary_confidence",
    # Temporal ordering
    "TemporalOrderer",
    "CardiacCycle",
    "assign_labels_by_intervals",
    "validate_segmentation",
    "SEGMENT_CLASSES",
    # Main segmenter
    "SelfSupervisedSegmenter",
    "SegmentationConfig",
    "create_segmenter",
]
