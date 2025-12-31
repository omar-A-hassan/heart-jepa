"""Unit tests for PCG preprocessing module."""

import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path
import scipy.io.wavfile as wavfile


class TestPreprocessing:
    """Tests for preprocessing functions."""

    @pytest.fixture
    def sample_pcg(self):
        """Generate synthetic PCG signal for testing."""
        sr = 2000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))

        # Simulate heart sounds with sinusoidal components
        # S1 at ~50Hz, S2 at ~100Hz
        s1_freq = 50
        s2_freq = 100
        heart_rate = 72  # bpm
        beat_interval = 60 / heart_rate

        pcg = np.zeros_like(t)

        # Add S1 and S2 sounds at regular intervals
        for i in range(int(duration / beat_interval)):
            beat_start = i * beat_interval
            s1_start = beat_start
            s2_start = beat_start + 0.3  # 300ms after S1

            # S1 envelope
            s1_mask = (t >= s1_start) & (t < s1_start + 0.1)
            pcg[s1_mask] += 0.8 * np.sin(2 * np.pi * s1_freq * (t[s1_mask] - s1_start))

            # S2 envelope
            s2_mask = (t >= s2_start) & (t < s2_start + 0.08)
            pcg[s2_mask] += 0.5 * np.sin(2 * np.pi * s2_freq * (t[s2_mask] - s2_start))

        # Add some noise
        pcg += 0.05 * np.random.randn(len(pcg))

        return pcg, sr

    @pytest.fixture
    def temp_wav_file(self, sample_pcg):
        """Create temporary wav file for testing."""
        pcg, sr = sample_pcg

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        # Scale to int16 range
        pcg_int16 = (pcg * 32767).astype(np.int16)
        wavfile.write(temp_path, sr, pcg_int16)

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_preprocess_pcg_shape(self, sample_pcg):
        """Test that preprocessing preserves signal length."""
        from heart_jepa.data.preprocessing import preprocess_pcg

        pcg, sr = sample_pcg
        pcg_processed = preprocess_pcg(pcg, sr)

        assert pcg_processed.shape == pcg.shape

    def test_preprocess_pcg_normalized(self, sample_pcg):
        """Test that preprocessing normalizes amplitude."""
        from heart_jepa.data.preprocessing import preprocess_pcg

        pcg, sr = sample_pcg
        pcg_processed = preprocess_pcg(pcg, sr, normalize=True)

        # Should be in [-1, 1] range
        assert np.max(np.abs(pcg_processed)) <= 1.0

    def test_preprocess_pcg_filtered(self, sample_pcg):
        """Test that bandpass filtering is applied."""
        from heart_jepa.data.preprocessing import preprocess_pcg

        pcg, sr = sample_pcg

        # Add high frequency noise that should be filtered
        pcg_noisy = pcg + 0.3 * np.sin(2 * np.pi * 1000 * np.arange(len(pcg)) / sr)

        pcg_processed = preprocess_pcg(pcg_noisy, sr, lowcut=20, highcut=500)

        # High frequency content should be reduced
        # Compare power in high frequency band
        from scipy import signal
        f, psd_noisy = signal.periodogram(pcg_noisy, sr)
        f, psd_processed = signal.periodogram(pcg_processed, sr)

        # Power above 500Hz should be much lower after filtering
        high_freq_mask = f > 600
        assert np.sum(psd_processed[high_freq_mask]) < 0.1 * np.sum(psd_noisy[high_freq_mask])

    def test_pcg_to_spectrogram_shape(self, sample_pcg):
        """Test spectrogram output shape."""
        from heart_jepa.data.preprocessing import pcg_to_spectrogram

        pcg, sr = sample_pcg
        spec = pcg_to_spectrogram(pcg, sr, n_mels=128)

        assert spec.shape[0] == 128  # n_mels
        assert spec.ndim == 2

    def test_pcg_to_spectrogram_values(self, sample_pcg):
        """Test spectrogram output values are reasonable."""
        from heart_jepa.data.preprocessing import pcg_to_spectrogram

        pcg, sr = sample_pcg
        spec = pcg_to_spectrogram(pcg, sr, to_db=True)

        # Should have finite values
        assert np.all(np.isfinite(spec))
        # dB values should be in reasonable range (not extremely large)
        assert np.max(spec) < 100  # Reasonable upper bound
        assert np.min(spec) > -100  # Reasonable lower bound

    def test_normalize_spectrogram_per_sample(self, sample_pcg):
        """Test per-sample spectrogram normalization."""
        from heart_jepa.data.preprocessing import pcg_to_spectrogram, normalize_spectrogram

        pcg, sr = sample_pcg
        spec = pcg_to_spectrogram(pcg, sr)
        spec_norm = normalize_spectrogram(spec, method="per_sample")

        # Mean should be close to 0
        assert np.abs(spec_norm.mean()) < 0.1
        # Std should be close to 1
        assert np.abs(spec_norm.std() - 1.0) < 0.1

    def test_normalize_spectrogram_minmax(self, sample_pcg):
        """Test min-max spectrogram normalization."""
        from heart_jepa.data.preprocessing import pcg_to_spectrogram, normalize_spectrogram

        pcg, sr = sample_pcg
        spec = pcg_to_spectrogram(pcg, sr)
        spec_norm = normalize_spectrogram(spec, method="minmax")

        # Should be in [0, 1] range
        assert spec_norm.min() >= 0
        assert spec_norm.max() <= 1

    def test_resize_spectrogram(self, sample_pcg):
        """Test spectrogram resizing."""
        from heart_jepa.data.preprocessing import pcg_to_spectrogram, resize_spectrogram

        pcg, sr = sample_pcg
        spec = pcg_to_spectrogram(pcg, sr, n_mels=128)

        # Resize to 224x224
        spec_resized = resize_spectrogram(spec, target_size=(224, 224))

        assert spec_resized.shape == (224, 224)

    def test_process_pcg_to_spectrogram_full_pipeline(self, sample_pcg):
        """Test full processing pipeline."""
        from heart_jepa.data.preprocessing import process_pcg_to_spectrogram

        pcg, sr = sample_pcg
        spec = process_pcg_to_spectrogram(pcg, sr)

        # Should be 224x224
        assert spec.shape == (224, 224)
        # Should have finite values
        assert np.all(np.isfinite(spec))

    def test_spectrogram_to_tensor(self, sample_pcg):
        """Test conversion to PyTorch tensor."""
        from heart_jepa.data.preprocessing import (
            process_pcg_to_spectrogram,
            spectrogram_to_tensor
        )

        pcg, sr = sample_pcg
        spec = process_pcg_to_spectrogram(pcg, sr)
        tensor = spectrogram_to_tensor(spec, add_channel_dim=True)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 224, 224)
        assert tensor.dtype == torch.float32

    def test_load_pcg_from_file(self, temp_wav_file):
        """Test loading PCG from wav file."""
        from heart_jepa.data.preprocessing import load_pcg

        pcg, sr = load_pcg(temp_wav_file, target_sr=2000)

        assert isinstance(pcg, np.ndarray)
        assert sr == 2000
        assert len(pcg) > 0

    def test_segment_pcg(self, sample_pcg):
        """Test PCG segmentation."""
        from heart_jepa.data.preprocessing import segment_pcg

        pcg, sr = sample_pcg
        segments = segment_pcg(pcg, sr, segment_duration=2.0, hop_duration=1.0)

        # Should have multiple segments
        assert len(segments) >= 1

        # Each segment should have correct length
        expected_length = int(2.0 * sr)
        for seg in segments:
            assert len(seg) == expected_length


class TestAugmentations:
    """Tests for augmentation functions."""

    @pytest.fixture
    def sample_spectrogram(self):
        """Generate sample spectrogram for testing."""
        spec = torch.randn(1, 224, 224)
        return spec

    @pytest.fixture
    def sample_pcg(self):
        """Generate sample PCG for testing."""
        sr = 2000
        duration = 5.0
        pcg = np.random.randn(int(sr * duration)) * 0.1
        return pcg, sr

    def test_spectrogram_augment_shape(self, sample_spectrogram):
        """Test that augmentation preserves shape."""
        from heart_jepa.data.augmentations import SpectrogramAugment

        aug = SpectrogramAugment()
        spec_aug = aug(sample_spectrogram)

        assert spec_aug.shape == sample_spectrogram.shape

    def test_spectrogram_augment_freq_mask(self, sample_spectrogram):
        """Test frequency masking."""
        from heart_jepa.data.augmentations import SpectrogramAugment

        # High probability to ensure mask is applied, disable noise to preserve zeros
        aug = SpectrogramAugment(p_freq_mask=1.0, n_freq_masks=2, p_time_mask=0.0, p_noise=0.0)
        spec_aug = aug(sample_spectrogram)

        # Some values should be zeroed
        assert (spec_aug == 0).any()

    def test_spectrogram_augment_time_mask(self, sample_spectrogram):
        """Test time masking."""
        from heart_jepa.data.augmentations import SpectrogramAugment

        # Disable noise to preserve zeros
        aug = SpectrogramAugment(p_time_mask=1.0, n_time_masks=2, p_freq_mask=0.0, p_noise=0.0)
        spec_aug = aug(sample_spectrogram)

        # Some values should be zeroed
        assert (spec_aug == 0).any()

    def test_pcg_augment_shape(self, sample_pcg):
        """Test that PCG augmentation returns similar length signal."""
        from heart_jepa.data.augmentations import PCGAugment

        pcg, sr = sample_pcg
        aug = PCGAugment(sr=sr)

        # Run multiple times to test stochasticity
        for _ in range(5):
            pcg_aug = aug(pcg)
            # Length may vary slightly due to time stretching
            assert len(pcg_aug) > 0
            # Should be clipped to [-1, 1]
            assert np.max(np.abs(pcg_aug)) <= 1.0

    def test_multi_view_transform(self, sample_pcg):
        """Test multi-view transform generates correct number of views."""
        from heart_jepa.data.augmentations import MultiViewTransform

        pcg, sr = sample_pcg
        transform = MultiViewTransform(n_views=4)
        views = transform(pcg, sr)

        assert views.shape[0] == 4  # 4 views
        assert views.shape[1] == 1  # 1 channel
        assert views.shape[2] == 224  # height
        assert views.shape[3] == 224  # width

    def test_test_transform(self, sample_pcg):
        """Test transform without augmentation."""
        from heart_jepa.data.augmentations import TestTransform

        pcg, sr = sample_pcg
        transform = TestTransform()
        spec = transform(pcg, sr)

        assert spec.shape == (1, 224, 224)
        assert isinstance(spec, torch.Tensor)


class TestDataset:
    """Tests for dataset classes."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with sample wav files."""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()

        # Create sample wav files
        sr = 2000
        duration = 5.0
        n_samples = int(sr * duration)

        for i in range(5):
            pcg = np.random.randn(n_samples).astype(np.float32) * 0.1
            pcg_int16 = (pcg * 32767).astype(np.int16)
            wav_path = os.path.join(temp_dir, f"sample_{i}.wav")
            wavfile.write(wav_path, sr, pcg_int16)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_pcg_dataset_length(self, temp_data_dir):
        """Test dataset length."""
        from heart_jepa.data.dataset import PCGDataset

        dataset = PCGDataset(temp_data_dir, split="train")
        assert len(dataset) == 5

    def test_pcg_dataset_getitem(self, temp_data_dir):
        """Test dataset getitem."""
        from heart_jepa.data.dataset import PCGDataset

        dataset = PCGDataset(temp_data_dir, split="train")
        spec, label = dataset[0]

        # Training mode returns multi-view
        assert spec.shape[0] == 4  # n_views
        assert spec.shape[1] == 1  # channels
        assert spec.shape[2] == 224  # height
        assert spec.shape[3] == 224  # width
        assert label == -1  # No labels provided

    def test_pcg_dataset_test_mode(self, temp_data_dir):
        """Test dataset in test mode (no augmentation)."""
        from heart_jepa.data.dataset import PCGDataset

        dataset = PCGDataset(temp_data_dir, split="test")
        spec, label = dataset[0]

        # Test mode returns single view
        assert spec.shape == (1, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
