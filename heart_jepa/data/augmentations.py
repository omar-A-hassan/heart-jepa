"""Augmentations for PCG signals and spectrograms."""

from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpectrogramAugment(nn.Module):
    """
    Spectrogram augmentations for multi-view learning (LEJEPA-style).

    Applies SpecAugment-style augmentations:
        - Time masking
        - Frequency masking
        - Time warping (optional)
        - Additive noise
    """

    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 30,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        noise_level: float = 0.01,
        p_freq_mask: float = 0.5,
        p_time_mask: float = 0.5,
        p_noise: float = 0.3,
    ):
        """
        Args:
            freq_mask_param: Maximum width of frequency mask
            time_mask_param: Maximum width of time mask
            n_freq_masks: Number of frequency masks to apply
            n_time_masks: Number of time masks to apply
            noise_level: Standard deviation of additive Gaussian noise
            p_freq_mask: Probability of applying frequency mask
            p_time_mask: Probability of applying time mask
            p_noise: Probability of adding noise
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.noise_level = noise_level
        self.p_freq_mask = p_freq_mask
        self.p_time_mask = p_time_mask
        self.p_noise = p_noise

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to spectrogram.

        Args:
            spec: Input spectrogram of shape (C, H, W) or (H, W)

        Returns:
            Augmented spectrogram
        """
        # Add batch dim if needed
        squeeze = False
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
            squeeze = True
        elif spec.dim() == 3:
            spec = spec.unsqueeze(0)

        spec = spec.clone()

        # Frequency masking
        if torch.rand(1).item() < self.p_freq_mask:
            spec = self._apply_freq_mask(spec)

        # Time masking
        if torch.rand(1).item() < self.p_time_mask:
            spec = self._apply_time_mask(spec)

        # Additive noise
        if torch.rand(1).item() < self.p_noise:
            noise = torch.randn_like(spec) * self.noise_level
            spec = spec + noise

        if squeeze:
            spec = spec.squeeze(0).squeeze(0)
        else:
            spec = spec.squeeze(0)

        return spec

    def _apply_freq_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking."""
        _, _, freq_dim, _ = spec.shape

        for _ in range(self.n_freq_masks):
            f = torch.randint(1, min(self.freq_mask_param, freq_dim) + 1, (1,)).item()
            f0 = torch.randint(0, max(1, freq_dim - f), (1,)).item()
            spec[:, :, f0:f0 + f, :] = 0

        return spec

    def _apply_time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply time masking."""
        _, _, _, time_dim = spec.shape

        for _ in range(self.n_time_masks):
            t = torch.randint(1, min(self.time_mask_param, time_dim) + 1, (1,)).item()
            t0 = torch.randint(0, max(1, time_dim - t), (1,)).item()
            spec[:, :, :, t0:t0 + t] = 0

        return spec


class PCGAugment:
    """
    Waveform-level augmentations for PCG signals.

    Applied before spectrogram conversion for multi-view generation.
    """

    def __init__(
        self,
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        gain_range: Tuple[float, float] = (-6, 6),
        noise_level: float = 0.005,
        p_time_stretch: float = 0.3,
        p_pitch_shift: float = 0.2,
        p_gain: float = 0.5,
        p_noise: float = 0.3,
        sr: int = 2000,
    ):
        """
        Args:
            time_stretch_range: Range for time stretching factor
            pitch_shift_range: Range for pitch shift in semitones
            gain_range: Range for gain in dB
            noise_level: Standard deviation of additive noise
            p_*: Probability of applying each augmentation
            sr: Sample rate
        """
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.gain_range = gain_range
        self.noise_level = noise_level
        self.p_time_stretch = p_time_stretch
        self.p_pitch_shift = p_pitch_shift
        self.p_gain = p_gain
        self.p_noise = p_noise
        self.sr = sr

    def __call__(self, pcg: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to PCG waveform.

        Args:
            pcg: Input PCG signal

        Returns:
            Augmented PCG signal
        """
        import librosa

        pcg = pcg.copy()

        # Time stretching
        if np.random.random() < self.p_time_stretch:
            rate = np.random.uniform(*self.time_stretch_range)
            pcg = librosa.effects.time_stretch(pcg, rate=rate)

        # Pitch shifting (careful with heart sounds!)
        if np.random.random() < self.p_pitch_shift:
            n_steps = np.random.randint(*self.pitch_shift_range)
            pcg = librosa.effects.pitch_shift(pcg, sr=self.sr, n_steps=n_steps)

        # Gain adjustment
        if np.random.random() < self.p_gain:
            gain_db = np.random.uniform(*self.gain_range)
            gain_linear = 10 ** (gain_db / 20)
            pcg = pcg * gain_linear

        # Additive noise
        if np.random.random() < self.p_noise:
            noise = np.random.randn(len(pcg)) * self.noise_level
            pcg = pcg + noise

        # Clip to prevent overflow
        pcg = np.clip(pcg, -1.0, 1.0)

        return pcg


class MultiViewTransform:
    """
    Generate multiple augmented views of a PCG signal for LEJEPA training.
    """

    def __init__(
        self,
        n_views: int = 4,
        pcg_augment: Optional[PCGAugment] = None,
        spec_augment: Optional[SpectrogramAugment] = None,
        config: Optional[dict] = None,
    ):
        """
        Args:
            n_views: Number of views to generate
            pcg_augment: PCG waveform augmentation (applied before spectrogram)
            spec_augment: Spectrogram augmentation
            config: Spectrogram configuration
        """
        self.n_views = n_views
        self.pcg_augment = pcg_augment or PCGAugment()
        self.spec_augment = spec_augment or SpectrogramAugment()
        self.config = config

    def __call__(
        self,
        pcg: np.ndarray,
        sr: int = 2000,
    ) -> torch.Tensor:
        """
        Generate multiple views of PCG signal.

        Args:
            pcg: Input PCG signal
            sr: Sample rate

        Returns:
            Tensor of shape (n_views, 1, H, W)
        """
        from .preprocessing import process_pcg_to_spectrogram, spectrogram_to_tensor

        views = []

        for _ in range(self.n_views):
            # Apply waveform augmentation
            pcg_aug = self.pcg_augment(pcg)

            # Convert to spectrogram
            spec = process_pcg_to_spectrogram(pcg_aug, sr, self.config)

            # Convert to tensor
            spec_tensor = spectrogram_to_tensor(spec)

            # Apply spectrogram augmentation
            spec_tensor = self.spec_augment(spec_tensor)

            views.append(spec_tensor)

        return torch.stack(views)  # (n_views, 1, H, W)


class TestTransform:
    """
    Transform for test/validation (no augmentation).
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config

    def __call__(self, pcg: np.ndarray, sr: int = 2000) -> torch.Tensor:
        """
        Convert PCG to spectrogram without augmentation.

        Args:
            pcg: Input PCG signal
            sr: Sample rate

        Returns:
            Tensor of shape (1, H, W)
        """
        from .preprocessing import process_pcg_to_spectrogram, spectrogram_to_tensor

        spec = process_pcg_to_spectrogram(pcg, sr, self.config)
        return spectrogram_to_tensor(spec)
