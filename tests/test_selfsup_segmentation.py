"""Tests for self-supervised segmentation module."""

import pytest
import torch
import numpy as np

from heart_jepa.segmentation import (
    # Attention
    get_attention_maps,
    attention_to_temporal,
    # Features
    FeatureExtractor,
    extract_patch_features,
    patches_to_temporal,
    get_frame_similarities,
    compute_feature_gradients,
    # Sound detection
    SoundDetector,
    SoundEvent,
    saliency_to_sound_mask,
    find_sound_events,
    # Clustering
    HeartSoundClusterer,
    assign_s1_s2_labels,
    # Boundary
    BoundaryDetector,
    refine_event_boundaries,
    # Temporal
    TemporalOrderer,
    assign_labels_by_intervals,
    validate_segmentation,
    SEGMENT_CLASSES,
    # Main
    SelfSupervisedSegmenter,
    SegmentationConfig,
)


class TestAttentionExtractor:
    """Tests for attention extraction."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        from heart_jepa.models import HeartJEPA
        return HeartJEPA(pretrained=False)

    @pytest.fixture
    def sample_input(self):
        """Create sample input."""
        return torch.randn(2, 1, 224, 224)

    def test_get_attention_maps_shape(self, model, sample_input):
        """Test attention map output shape."""
        attn = get_attention_maps(model, sample_input, layer_idx=-1, aggregate="mean")

        # Should be (B, N, N) where N = 197 (196 patches + 1 CLS)
        assert attn.shape == (2, 197, 197)

    def test_get_attention_maps_no_aggregate(self, model, sample_input):
        """Test attention without aggregation."""
        attn = get_attention_maps(model, sample_input, layer_idx=-1, aggregate=None)

        # Should be (B, num_heads, N, N)
        assert attn.dim() == 4
        assert attn.shape[0] == 2
        assert attn.shape[2] == 197
        assert attn.shape[3] == 197

    def test_attention_to_temporal(self, model, sample_input):
        """Test conversion to temporal saliency."""
        attn = get_attention_maps(model, sample_input)
        saliency = attention_to_temporal(attn, output_frames=224)

        assert saliency.shape == (2, 224)

    def test_attention_values_normalized(self, model, sample_input):
        """Test that attention values sum to 1."""
        attn = get_attention_maps(model, sample_input, aggregate="mean")

        # Each row should sum to ~1 (softmax output)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestFeatureExtractor:
    """Tests for feature extraction."""

    @pytest.fixture
    def model(self):
        from heart_jepa.models import HeartJEPA
        return HeartJEPA(pretrained=False)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 1, 224, 224)

    def test_extract_patch_features_shape(self, model, sample_input):
        """Test patch feature extraction."""
        features = extract_patch_features(model, sample_input)

        # Should be (B, N+1, embed_dim) where N=196
        assert features.shape == (2, 197, 768)

    def test_feature_extractor_output_shape(self, model, sample_input):
        """Test FeatureExtractor output."""
        extractor = FeatureExtractor(model, output_frames=224)
        features, cls_token = extractor.extract(sample_input, return_cls=True)

        assert features.shape == (2, 224, 768)
        assert cls_token.shape == (2, 768)

    def test_patches_to_temporal(self):
        """Test patch to temporal conversion."""
        # 196 patches (14x14 grid)
        patches = torch.randn(2, 196, 768)
        temporal = patches_to_temporal(patches, output_frames=224)

        assert temporal.shape == (2, 224, 768)

    def test_get_frame_similarities(self):
        """Test frame similarity computation."""
        features = torch.randn(2, 50, 128)
        similarities = get_frame_similarities(features, method="cosine")

        assert similarities.shape == (2, 50, 50)
        # Diagonal should be 1 (self-similarity)
        diag = torch.diagonal(similarities, dim1=1, dim2=2)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)

    def test_compute_feature_gradients(self):
        """Test gradient computation."""
        features = torch.randn(2, 50, 128)
        gradients = compute_feature_gradients(features)

        assert gradients.shape == (2, 49)  # T-1
        assert (gradients >= 0).all()  # Non-negative


class TestSoundDetector:
    """Tests for sound detection."""

    @pytest.fixture
    def sample_saliency(self):
        """Create sample saliency with clear peaks."""
        # Create saliency with 4 clear peaks
        saliency = torch.zeros(2, 224)
        for b in range(2):
            for i in range(4):
                peak = 50 + i * 50
                peak_values = torch.cat([
                    torch.linspace(0.2, 1.0, 5),
                    torch.linspace(1.0, 0.2, 5)
                ])
                saliency[b, peak-5:peak+5] = peak_values
        return saliency

    def test_sound_detector_basic(self, sample_saliency):
        """Test basic sound detection."""
        detector = SoundDetector(threshold=0.3, min_duration=3)
        mask, events = detector.detect(sample_saliency)

        assert mask.shape == (2, 224)
        assert len(events) == 2  # Two samples
        # Should detect approximately 4 events per sample
        assert len(events[0]) >= 2

    def test_sound_event_properties(self, sample_saliency):
        """Test SoundEvent dataclass."""
        detector = SoundDetector()
        _, events = detector.detect(sample_saliency)

        if events[0]:
            event = events[0][0]
            assert event.start >= 0
            assert event.end > event.start
            assert event.start <= event.peak <= event.end
            assert event.duration == event.end - event.start

    def test_saliency_to_sound_mask(self, sample_saliency):
        """Test convenience function."""
        mask = saliency_to_sound_mask(sample_saliency, threshold=0.3)
        assert mask.dtype == torch.bool
        assert mask.shape == sample_saliency.shape

    def test_find_sound_events(self, sample_saliency):
        """Test convenience function."""
        events = find_sound_events(sample_saliency)
        assert isinstance(events, list)
        assert len(events) == 2


class TestClustering:
    """Tests for clustering."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features with two clusters."""
        features = torch.randn(224, 128)
        # Make first half and second half different
        features[:112] += 1.0
        features[112:] -= 1.0
        return features

    @pytest.fixture
    def sample_events(self):
        """Create sample events."""
        return [
            SoundEvent(start=10, end=20, peak=15, intensity=0.8),
            SoundEvent(start=60, end=70, peak=65, intensity=0.9),
            SoundEvent(start=110, end=120, peak=115, intensity=0.7),
            SoundEvent(start=160, end=170, peak=165, intensity=0.85),
        ]

    def test_cluster_events_kmeans(self, sample_features, sample_events):
        """Test K-means clustering."""
        clusterer = HeartSoundClusterer(method="kmeans", n_clusters=2)
        labels = clusterer.cluster_events(sample_features, sample_events)

        assert len(labels) == len(sample_events)
        assert set(labels) <= {0, 1}

    def test_cluster_events_gmm(self, sample_features, sample_events):
        """Test GMM clustering."""
        clusterer = HeartSoundClusterer(method="gmm", n_clusters=2)
        labels = clusterer.cluster_events(sample_features, sample_events)

        assert len(labels) == len(sample_events)

    def test_assign_s1_s2_labels(self, sample_events):
        """Test S1/S2 label assignment."""
        cluster_labels = [0, 1, 0, 1]  # Alternating clusters
        s1_indices, s2_indices = assign_s1_s2_labels(sample_events, cluster_labels)

        assert len(s1_indices) + len(s2_indices) == len(sample_events)
        assert set(s1_indices).isdisjoint(set(s2_indices))

    def test_empty_events(self, sample_features):
        """Test with no events."""
        clusterer = HeartSoundClusterer()
        labels = clusterer.cluster_events(sample_features, [])
        assert labels == []


class TestBoundaryDetector:
    """Tests for boundary detection."""

    @pytest.fixture
    def sample_features(self):
        """Create features with clear boundaries."""
        features = torch.zeros(100, 64)
        # Create 3 segments with different distributions
        features[:30] = torch.randn(30, 64) + 0
        features[30:70] = torch.randn(40, 64) + 2
        features[70:] = torch.randn(30, 64) - 2
        return features

    def test_detect_boundaries_gradient(self, sample_features):
        """Test gradient-based boundary detection."""
        detector = BoundaryDetector(method="gradient", min_segment_length=10)
        boundaries = detector.detect_boundaries(sample_features)

        assert isinstance(boundaries, list)
        # Should find boundaries around 30 and 70
        assert len(boundaries) >= 1

    def test_refine_event_boundaries(self, sample_features):
        """Test boundary refinement."""
        events = [
            SoundEvent(start=25, end=35, peak=30, intensity=0.8),
            SoundEvent(start=65, end=75, peak=70, intensity=0.9),
        ]
        refined = refine_event_boundaries(events, sample_features)

        assert len(refined) == len(events)
        for e in refined:
            assert e.end > e.start


class TestTemporalOrdering:
    """Tests for temporal ordering."""

    @pytest.fixture
    def sample_events(self):
        """Create events simulating cardiac cycle."""
        # S1 at 20, S2 at 40, S1 at 100, S2 at 120
        return [
            SoundEvent(start=15, end=25, peak=20, intensity=0.9),
            SoundEvent(start=35, end=45, peak=40, intensity=0.8),
            SoundEvent(start=95, end=105, peak=100, intensity=0.9),
            SoundEvent(start=115, end=125, peak=120, intensity=0.8),
        ]

    def test_assign_labels_by_intervals(self, sample_events):
        """Test interval-based labeling."""
        labels = assign_labels_by_intervals(sample_events, total_frames=224)

        assert labels.shape == (224,)
        assert labels.dtype == np.int64

        # Should have S1, S2, systole, diastole
        unique_labels = set(labels.tolist())
        assert SEGMENT_CLASSES["S1"] in unique_labels
        assert SEGMENT_CLASSES["S2"] in unique_labels

    def test_temporal_orderer_order_events(self, sample_events):
        """Test TemporalOrderer."""
        orderer = TemporalOrderer()
        s1_indices = [0, 2]
        s2_indices = [1, 3]

        cycles = orderer.order_events(sample_events, s1_indices, s2_indices)

        assert len(cycles) >= 1
        for cycle in cycles:
            if cycle.s1_event and cycle.s2_event:
                assert cycle.s1_event.start < cycle.s2_event.start

    def test_validate_segmentation_valid(self):
        """Test validation with valid segmentation."""
        # Create valid labels: alternating S1-systole-S2-diastole
        labels = np.zeros(224, dtype=np.int64)
        for i in range(4):
            base = i * 50
            labels[base:base+10] = SEGMENT_CLASSES["S1"]
            labels[base+10:base+20] = SEGMENT_CLASSES["systole"]
            labels[base+20:base+30] = SEGMENT_CLASSES["S2"]
            labels[base+30:base+50] = SEGMENT_CLASSES["diastole"]

        is_valid, msg = validate_segmentation(labels)
        assert is_valid

    def test_validate_segmentation_invalid(self):
        """Test validation with invalid segmentation."""
        # All S1, no S2
        labels = np.zeros(224, dtype=np.int64)
        labels[10:20] = SEGMENT_CLASSES["S1"]

        is_valid, msg = validate_segmentation(labels)
        assert not is_valid


class TestSelfSupervisedSegmenter:
    """Tests for main segmenter class."""

    @pytest.fixture
    def model(self):
        from heart_jepa.models import HeartJEPA
        return HeartJEPA(pretrained=False)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 1, 224, 224)

    def test_segmenter_creation(self, model):
        """Test segmenter instantiation."""
        config = SegmentationConfig(output_frames=224)
        segmenter = SelfSupervisedSegmenter(model, config)

        assert segmenter.config.output_frames == 224

    def test_segment_single(self, model):
        """Test segmentation of single sample."""
        segmenter = SelfSupervisedSegmenter(model)
        spec = torch.randn(1, 224, 224)

        labels = segmenter.segment(spec)

        assert labels.shape == (224,)
        assert labels.dtype == torch.int64

    def test_segment_batch(self, model, sample_input):
        """Test batch segmentation."""
        segmenter = SelfSupervisedSegmenter(model)
        labels = segmenter.segment_batch(sample_input)

        assert labels.shape == (2, 224)

    def test_segment_with_intermediate(self, model):
        """Test segmentation with intermediate results."""
        segmenter = SelfSupervisedSegmenter(model)
        spec = torch.randn(1, 1, 224, 224)

        labels, intermediate = segmenter.segment(spec, return_intermediate=True)

        assert len(intermediate) == 1
        assert 'saliency' in intermediate[0]
        assert 'sound_mask' in intermediate[0]
        assert 'events' in intermediate[0]

    def test_get_config(self, model):
        """Test config retrieval."""
        config = SegmentationConfig(min_hr=50, max_hr=180)
        segmenter = SelfSupervisedSegmenter(model, config)

        cfg = segmenter.get_config()
        assert cfg['min_hr'] == 50
        assert cfg['max_hr'] == 180


class TestIntegration:
    """Integration tests for full pipeline."""

    @pytest.fixture
    def model(self):
        from heart_jepa.models import HeartJEPA
        return HeartJEPA(pretrained=False)

    def test_full_pipeline(self, model):
        """Test complete segmentation pipeline."""
        # Create synthetic spectrogram
        spec = torch.randn(1, 1, 224, 224)

        # Run segmentation
        segmenter = SelfSupervisedSegmenter(model)
        labels = segmenter.segment(spec)

        # Verify output
        assert labels.shape == (224,)
        assert set(labels.numpy().tolist()).issubset(set(SEGMENT_CLASSES.values()))

    def test_pipeline_deterministic(self, model):
        """Test that pipeline is deterministic."""
        torch.manual_seed(42)
        spec = torch.randn(1, 1, 224, 224)

        segmenter = SelfSupervisedSegmenter(model)

        labels1 = segmenter.segment(spec.clone())
        labels2 = segmenter.segment(spec.clone())

        assert torch.equal(labels1, labels2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
