"""Boundary detection for segment refinement."""

import torch
import numpy as np
from typing import List, Tuple, Optional
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from .sound_detector import SoundEvent


class BoundaryDetector:
    """
    Detect segment boundaries using change-point detection.

    Refines segment boundaries by detecting transitions in
    the feature space.
    """

    def __init__(
        self,
        method: str = "gradient",
        min_segment_length: int = 5,
        smooth_sigma: float = 2.0,
        peak_prominence: float = 0.1,
    ):
        """
        Initialize boundary detector.

        Args:
            method: Detection method ("gradient", "pelt", "cusum")
            min_segment_length: Minimum frames between boundaries
            smooth_sigma: Gaussian smoothing sigma for gradient method
            peak_prominence: Minimum prominence for peak detection
        """
        self.method = method
        self.min_segment_length = min_segment_length
        self.smooth_sigma = smooth_sigma
        self.peak_prominence = peak_prominence

    def detect_boundaries(
        self,
        features: torch.Tensor,
    ) -> List[int]:
        """
        Detect boundaries in feature sequence.

        Args:
            features: Frame features of shape (T, D) or (B, T, D)

        Returns:
            List of boundary frame indices
        """
        if features.dim() == 3:
            # Batch mode - process first sample
            features = features[0]

        features_np = features.cpu().numpy() if torch.is_tensor(features) else features

        if self.method == "gradient":
            boundaries = self._detect_gradient_boundaries(features_np)
        elif self.method == "pelt":
            boundaries = self._detect_pelt_boundaries(features_np)
        elif self.method == "cusum":
            boundaries = self._detect_cusum_boundaries(features_np)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Filter by minimum segment length
        boundaries = self._filter_boundaries(boundaries, len(features_np))

        return boundaries

    def _detect_gradient_boundaries(self, features: np.ndarray) -> List[int]:
        """Detect boundaries using feature gradient magnitude."""
        T, D = features.shape

        # Compute gradient magnitude
        diff = np.diff(features, axis=0)  # (T-1, D)
        gradient_mag = np.linalg.norm(diff, axis=1)  # (T-1,)

        # Smooth
        if self.smooth_sigma > 0:
            gradient_mag = gaussian_filter1d(gradient_mag, sigma=self.smooth_sigma)

        # Normalize
        gradient_mag = gradient_mag / (gradient_mag.max() + 1e-8)

        # Find peaks (boundaries are where gradient is high)
        peaks, properties = find_peaks(
            gradient_mag,
            prominence=self.peak_prominence,
            distance=self.min_segment_length,
        )

        # Adjust indices (diff reduces length by 1)
        boundaries = (peaks + 1).tolist()

        return boundaries

    def _detect_pelt_boundaries(self, features: np.ndarray) -> List[int]:
        """Detect boundaries using PELT algorithm from ruptures."""
        try:
            import ruptures as rpt
        except ImportError:
            # Fall back to gradient method
            return self._detect_gradient_boundaries(features)

        T, D = features.shape

        # Use RBF kernel cost function
        model = "rbf"
        algo = rpt.Pelt(model=model, min_size=self.min_segment_length).fit(features)

        # Predict with automatic penalty selection
        penalty = np.log(T) * D  # BIC-like penalty
        boundaries = algo.predict(pen=penalty)

        # Remove last boundary (always T)
        if boundaries and boundaries[-1] == T:
            boundaries = boundaries[:-1]

        return boundaries

    def _detect_cusum_boundaries(self, features: np.ndarray) -> List[int]:
        """Detect boundaries using CUSUM algorithm."""
        T, D = features.shape

        # Compute cumulative sum of deviations from mean
        mean_feat = features.mean(axis=0)
        deviations = features - mean_feat
        cusum = np.cumsum(np.linalg.norm(deviations, axis=1))

        # Detrend
        trend = np.linspace(0, cusum[-1], T)
        cusum_detrended = cusum - trend

        # Find peaks (change points are extrema of detrended CUSUM)
        peaks, _ = find_peaks(
            np.abs(cusum_detrended),
            prominence=self.peak_prominence * cusum_detrended.max(),
            distance=self.min_segment_length,
        )

        return peaks.tolist()

    def _filter_boundaries(
        self,
        boundaries: List[int],
        total_length: int,
    ) -> List[int]:
        """Filter boundaries by minimum segment length."""
        if not boundaries:
            return []

        # Sort boundaries
        boundaries = sorted(boundaries)

        # Add implicit boundaries at start and end
        all_points = [0] + boundaries + [total_length]

        # Filter
        filtered = []
        prev = 0
        for b in boundaries:
            if b - prev >= self.min_segment_length:
                if total_length - b >= self.min_segment_length:
                    filtered.append(b)
                    prev = b

        return filtered


def refine_event_boundaries(
    events: List[SoundEvent],
    features: torch.Tensor,
    boundary_detector: Optional[BoundaryDetector] = None,
    search_window: int = 5,
) -> List[SoundEvent]:
    """
    Refine sound event boundaries using feature gradients.

    Args:
        events: List of SoundEvent objects
        features: Frame features of shape (T, D)
        boundary_detector: Optional pre-configured detector
        search_window: Frames to search around current boundary

    Returns:
        List of refined SoundEvent objects
    """
    if not events:
        return []

    if boundary_detector is None:
        boundary_detector = BoundaryDetector()

    features_np = features.cpu().numpy() if torch.is_tensor(features) else features
    T, D = features_np.shape

    # Compute gradient magnitude for the whole signal
    diff = np.diff(features_np, axis=0)
    gradient_mag = np.linalg.norm(diff, axis=1)
    gradient_mag = gaussian_filter1d(gradient_mag, sigma=1.0)

    refined_events = []
    for event in events:
        # Refine start boundary
        search_start = max(0, event.start - search_window)
        search_end = min(T - 1, event.start + search_window)

        if search_end > search_start:
            local_gradient = gradient_mag[search_start:search_end]
            refined_start = search_start + int(np.argmax(local_gradient))
        else:
            refined_start = event.start

        # Refine end boundary
        search_start = max(0, event.end - search_window)
        search_end = min(T - 1, event.end + search_window)

        if search_end > search_start:
            local_gradient = gradient_mag[search_start:search_end]
            refined_end = search_start + int(np.argmax(local_gradient))
        else:
            refined_end = event.end

        # Ensure valid event
        if refined_end <= refined_start:
            refined_end = refined_start + 1

        refined_events.append(SoundEvent(
            start=refined_start,
            end=refined_end,
            peak=event.peak,
            intensity=event.intensity,
        ))

    return refined_events


def compute_boundary_confidence(
    features: torch.Tensor,
    boundaries: List[int],
    window: int = 3,
) -> List[float]:
    """
    Compute confidence scores for detected boundaries.

    Args:
        features: Frame features of shape (T, D)
        boundaries: List of boundary indices
        window: Window size for gradient computation

    Returns:
        List of confidence scores (0-1) for each boundary
    """
    if not boundaries:
        return []

    features_np = features.cpu().numpy() if torch.is_tensor(features) else features
    T, D = features_np.shape

    # Compute gradient magnitude
    diff = np.diff(features_np, axis=0)
    gradient_mag = np.linalg.norm(diff, axis=1)

    # Normalize
    max_grad = gradient_mag.max() + 1e-8

    confidences = []
    for b in boundaries:
        # Get gradient at boundary
        start = max(0, b - window)
        end = min(len(gradient_mag), b + window)

        if end > start:
            local_max = gradient_mag[start:end].max()
            confidence = local_max / max_grad
        else:
            confidence = 0.5

        confidences.append(float(confidence))

    return confidences
