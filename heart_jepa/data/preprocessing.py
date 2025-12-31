"""PCG signal preprocessing using neurokit2 and librosa."""

from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import librosa
import neurokit2 as nk
import scipy.io.wavfile as wavfile


# Default spectrogram configuration optimized for PCG
SPECTROGRAM_CONFIG = {
    "sample_rate": 2000,      # PhysioNet standard
    "duration": 5.0,          # seconds (~5-8 heartbeats)
    "n_fft": 512,             # ~4 Hz frequency resolution
    "hop_length": 64,         # ~32ms time resolution
    "n_mels": 128,            # mel frequency bins
    "fmin": 20,               # Hz - captures S3/S4
    "fmax": 500,              # Hz - captures murmurs
    "image_size": 224,        # resize to ViT input
}


def load_pcg(
    path: Union[str, Path],
    target_sr: int = 2000,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load PCG audio file and resample to target sample rate.

    Args:
        path: Path to audio file (wav, mp3, etc.)
        target_sr: Target sample rate (default: 2000 Hz for PhysioNet)
        duration: Optional duration in seconds to load

    Returns:
        Tuple of (audio array, sample rate)
    """
    path = Path(path)

    # Load with librosa (handles resampling)
    pcg, sr = librosa.load(str(path), sr=target_sr, duration=duration, mono=True)

    return pcg, sr


def preprocess_pcg(
    pcg: np.ndarray,
    sr: int = 2000,
    lowcut: float = 20.0,
    highcut: float = 500.0,
    remove_drift: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess PCG signal using neurokit2.

    Pipeline:
        1. Bandpass filter (20-500 Hz for heart sounds)
        2. Remove baseline drift (polynomial detrending)
        3. Normalize amplitude

    Args:
        pcg: Raw PCG signal
        sr: Sample rate
        lowcut: Low frequency cutoff (Hz)
        highcut: High frequency cutoff (Hz)
        remove_drift: Whether to remove baseline drift
        normalize: Whether to normalize amplitude to [-1, 1]

    Returns:
        Preprocessed PCG signal
    """
    # Bandpass filter using neurokit2
    pcg_filtered = nk.signal_filter(
        pcg,
        sampling_rate=sr,
        lowcut=lowcut,
        highcut=highcut,
        method='butterworth',
        order=4
    )

    # Remove baseline drift
    if remove_drift:
        pcg_filtered = nk.signal_detrend(
            pcg_filtered,
            method='polynomial',
            order=1
        )

    # Normalize amplitude
    if normalize:
        max_val = np.max(np.abs(pcg_filtered))
        if max_val > 0:
            pcg_filtered = pcg_filtered / max_val

    return pcg_filtered


def pcg_to_spectrogram(
    pcg: np.ndarray,
    sr: int = 2000,
    n_fft: int = 512,
    hop_length: int = 64,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float = 500.0,
    power: float = 2.0,
    to_db: bool = True,
    ref: float = 1.0,
) -> np.ndarray:
    """
    Convert PCG signal to mel-spectrogram.

    Args:
        pcg: Preprocessed PCG signal
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length between frames
        n_mels: Number of mel frequency bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        power: Exponent for magnitude spectrogram
        to_db: Convert to decibel scale
        ref: Reference value for dB conversion

    Returns:
        Mel-spectrogram array of shape (n_mels, time_frames)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=pcg,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )

    # Convert to dB scale
    if to_db:
        mel_spec = librosa.power_to_db(mel_spec, ref=ref)

    return mel_spec


def normalize_spectrogram(
    spec: np.ndarray,
    method: str = "per_sample",
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> np.ndarray:
    """
    Normalize spectrogram values.

    Args:
        spec: Input spectrogram
        method: Normalization method
            - "per_sample": Normalize each spectrogram independently
            - "global": Use provided mean/std
            - "minmax": Scale to [0, 1]
        mean: Global mean (for method="global")
        std: Global std (for method="global")

    Returns:
        Normalized spectrogram
    """
    if method == "per_sample":
        mean = spec.mean()
        std = spec.std()
        if std > 0:
            spec = (spec - mean) / std
        else:
            spec = spec - mean

    elif method == "global":
        if mean is None or std is None:
            raise ValueError("mean and std required for global normalization")
        spec = (spec - mean) / std

    elif method == "minmax":
        min_val = spec.min()
        max_val = spec.max()
        if max_val > min_val:
            spec = (spec - min_val) / (max_val - min_val)
        else:
            spec = spec - min_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return spec


def resize_spectrogram(
    spec: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Resize spectrogram to target size for ViT input.

    Args:
        spec: Input spectrogram of shape (freq, time)
        target_size: Target (height, width) for output

    Returns:
        Resized spectrogram
    """
    import cv2

    # cv2.resize expects (width, height)
    resized = cv2.resize(spec, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return resized


def process_pcg_to_spectrogram(
    pcg: np.ndarray,
    sr: int = 2000,
    config: Optional[dict] = None,
) -> np.ndarray:
    """
    Full pipeline: preprocess PCG and convert to normalized spectrogram.

    Args:
        pcg: Raw PCG signal
        sr: Sample rate
        config: Optional config dict (defaults to SPECTROGRAM_CONFIG)

    Returns:
        Normalized spectrogram of shape (224, 224)
    """
    config = config or SPECTROGRAM_CONFIG

    # Preprocess
    pcg_clean = preprocess_pcg(
        pcg, sr,
        lowcut=config.get("fmin", 20),
        highcut=config.get("fmax", 500),
    )

    # Convert to spectrogram
    spec = pcg_to_spectrogram(
        pcg_clean, sr,
        n_fft=config.get("n_fft", 512),
        hop_length=config.get("hop_length", 64),
        n_mels=config.get("n_mels", 128),
        fmin=config.get("fmin", 20),
        fmax=config.get("fmax", 500),
    )

    # Normalize
    spec = normalize_spectrogram(spec, method="per_sample")

    # Resize to target size
    target_size = config.get("image_size", 224)
    spec = resize_spectrogram(spec, (target_size, target_size))

    return spec


def spectrogram_to_tensor(
    spec: np.ndarray,
    add_channel_dim: bool = True,
) -> torch.Tensor:
    """
    Convert spectrogram numpy array to PyTorch tensor.

    Args:
        spec: Spectrogram array
        add_channel_dim: Whether to add channel dimension

    Returns:
        Tensor of shape (1, H, W) if add_channel_dim else (H, W)
    """
    tensor = torch.from_numpy(spec).float()
    if add_channel_dim:
        tensor = tensor.unsqueeze(0)
    return tensor


def segment_pcg(
    pcg: np.ndarray,
    sr: int = 2000,
    segment_duration: float = 5.0,
    hop_duration: float = 2.5,
    min_duration: float = 3.0,
) -> list:
    """
    Segment long PCG recording into fixed-length segments.

    Args:
        pcg: PCG signal
        sr: Sample rate
        segment_duration: Duration of each segment in seconds
        hop_duration: Hop between segments in seconds
        min_duration: Minimum duration for last segment

    Returns:
        List of PCG segments
    """
    segment_samples = int(segment_duration * sr)
    hop_samples = int(hop_duration * sr)
    min_samples = int(min_duration * sr)

    segments = []
    start = 0

    while start + min_samples <= len(pcg):
        end = min(start + segment_samples, len(pcg))
        segment = pcg[start:end]

        # Pad if necessary
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')

        segments.append(segment)
        start += hop_samples

    return segments
