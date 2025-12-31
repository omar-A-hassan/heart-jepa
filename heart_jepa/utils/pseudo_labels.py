"""Pseudo-label generation for heart sound segmentation."""

import numpy as np
from typing import Tuple, Dict, Optional, List
import neurokit2 as nk


# Class indices for segmentation
SEGMENT_CLASSES = {
    "background": 0,
    "S1": 1,
    "systole": 2,
    "S2": 3,
    "diastole": 4,
    "S3": 5,
    "S4": 6,
}


def shannon_envelope(signal: np.ndarray, window_size: int = 40) -> np.ndarray:
    """
    Compute Shannon energy envelope of PCG signal.

    Shannon energy is sensitive to medium-intensity sounds (like heart sounds)
    while being less affected by high-amplitude noise.

    Args:
        signal: Input PCG signal
        window_size: Window size for smoothing (in samples)

    Returns:
        Shannon energy envelope
    """
    # Normalize signal
    signal = signal / (np.max(np.abs(signal)) + 1e-10)

    # Shannon energy: -x^2 * log(x^2)
    # Handle zeros and very small values
    eps = 1e-10
    x2 = signal ** 2 + eps
    shannon = -x2 * np.log(x2)

    # Smooth with moving average
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        shannon = np.convolve(shannon, kernel, mode="same")

    return shannon


def detect_heart_sounds(
    pcg: np.ndarray,
    sr: int = 2000,
    method: str = "envelope",
    min_peak_distance: float = 0.2,
    s1_duration: float = 0.12,
    s2_duration: float = 0.10,
    expected_hr_range: Tuple[float, float] = (40, 200),
) -> Dict[str, np.ndarray]:
    """
    Detect S1 and S2 heart sounds from PCG signal.

    Uses envelope-based peak detection with physiological constraints.

    Args:
        pcg: Input PCG signal (preprocessed)
        sr: Sample rate
        method: Detection method ('envelope', 'hilbert', 'shannon')
        min_peak_distance: Minimum distance between peaks in seconds
        s1_duration: Expected S1 duration in seconds
        s2_duration: Expected S2 duration in seconds
        expected_hr_range: Expected heart rate range in bpm

    Returns:
        Dictionary containing:
        - 's1_peaks': S1 peak locations (sample indices)
        - 's2_peaks': S2 peak locations (sample indices)
        - 's1_onsets': S1 onset locations
        - 's1_offsets': S1 offset locations
        - 's2_onsets': S2 onset locations
        - 's2_offsets': S2 offset locations
        - 'envelope': Computed envelope signal
    """
    # Compute envelope
    if method == "shannon":
        envelope = shannon_envelope(pcg, window_size=int(0.02 * sr))
    elif method == "hilbert":
        from scipy.signal import hilbert
        analytic = np.abs(hilbert(pcg))
        envelope = nk.signal_smooth(analytic, method="convolution", kernel="boxcar", size=int(0.02 * sr))
    else:  # envelope
        envelope = np.abs(pcg)
        envelope = nk.signal_smooth(envelope, method="convolution", kernel="boxcar", size=int(0.02 * sr))

    # Normalize envelope
    envelope = envelope / (np.max(envelope) + 1e-10)

    # Find peaks
    min_distance_samples = int(min_peak_distance * sr)

    # Find all significant peaks
    peaks_info = nk.signal_findpeaks(
        envelope,
        height_min=0.1,
    )
    peaks = peaks_info["Peaks"]

    # Filter peaks by minimum distance
    if len(peaks) > 1:
        peaks = _filter_peaks_by_distance(peaks, min_distance_samples)

    if len(peaks) < 2:
        # Not enough peaks found
        return {
            "s1_peaks": np.array([]),
            "s2_peaks": np.array([]),
            "s1_onsets": np.array([]),
            "s1_offsets": np.array([]),
            "s2_onsets": np.array([]),
            "s2_offsets": np.array([]),
            "envelope": envelope,
        }

    # Classify peaks as S1 or S2 based on intervals
    # S1-S2 interval (systole) is shorter than S2-S1 interval (diastole)
    s1_peaks, s2_peaks = _classify_heart_sounds(peaks, sr, expected_hr_range)

    # Estimate onset/offset for each heart sound
    s1_onsets, s1_offsets = _estimate_boundaries(
        peaks=s1_peaks,
        envelope=envelope,
        duration=s1_duration,
        sr=sr,
    )

    s2_onsets, s2_offsets = _estimate_boundaries(
        peaks=s2_peaks,
        envelope=envelope,
        duration=s2_duration,
        sr=sr,
    )

    return {
        "s1_peaks": s1_peaks,
        "s2_peaks": s2_peaks,
        "s1_onsets": s1_onsets,
        "s1_offsets": s1_offsets,
        "s2_onsets": s2_onsets,
        "s2_offsets": s2_offsets,
        "envelope": envelope,
    }


def _filter_peaks_by_distance(
    peaks: np.ndarray,
    min_distance: int,
) -> np.ndarray:
    """
    Filter peaks to ensure minimum distance between consecutive peaks.

    Keeps the first peak and removes subsequent peaks that are too close.
    """
    if len(peaks) == 0:
        return peaks

    filtered = [peaks[0]]
    for peak in peaks[1:]:
        if peak - filtered[-1] >= min_distance:
            filtered.append(peak)

    return np.array(filtered)


def _classify_heart_sounds(
    peaks: np.ndarray,
    sr: int,
    expected_hr_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify peaks as S1 or S2 based on interval patterns.

    The systolic interval (S1-S2) is typically shorter than the diastolic
    interval (S2-S1). Uses this pattern to distinguish S1 from S2.
    """
    if len(peaks) < 2:
        return np.array([]), np.array([])

    # Compute intervals between consecutive peaks
    intervals = np.diff(peaks) / sr  # in seconds

    # Expected beat duration range
    min_beat_dur = 60 / expected_hr_range[1]  # seconds per beat at max HR
    max_beat_dur = 60 / expected_hr_range[0]  # seconds per beat at min HR

    # Systole is typically 0.3-0.4 of the cardiac cycle
    # Use alternating pattern: short (systole) - long (diastole)

    # Find pairs with short-long pattern
    s1_peaks = []
    s2_peaks = []

    # Start with first peak, determine if it's S1 or S2
    # by looking at the interval pattern
    if len(intervals) >= 2:
        # If first interval is shorter than second, first peak is likely S1
        if intervals[0] < intervals[1]:
            start_with_s1 = True
        else:
            start_with_s1 = False
    else:
        # Default to S1
        start_with_s1 = True

    for i, peak in enumerate(peaks):
        if start_with_s1:
            if i % 2 == 0:
                s1_peaks.append(peak)
            else:
                s2_peaks.append(peak)
        else:
            if i % 2 == 0:
                s2_peaks.append(peak)
            else:
                s1_peaks.append(peak)

    return np.array(s1_peaks), np.array(s2_peaks)


def _estimate_boundaries(
    peaks: np.ndarray,
    envelope: np.ndarray,
    duration: float,
    sr: int,
    threshold_ratio: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate onset and offset of heart sounds based on envelope.

    Uses threshold-based detection around each peak.
    """
    onsets = []
    offsets = []

    half_duration = int(duration * sr / 2)

    for peak in peaks:
        # Get local envelope around peak
        start = max(0, peak - half_duration * 2)
        end = min(len(envelope), peak + half_duration * 2)
        local_env = envelope[start:end]
        local_peak = peak - start

        if len(local_env) == 0:
            continue

        # Threshold
        threshold = threshold_ratio * local_env[local_peak]

        # Find onset (search backward from peak)
        onset = local_peak
        for i in range(local_peak - 1, -1, -1):
            if local_env[i] < threshold:
                onset = i
                break

        # Find offset (search forward from peak)
        offset = local_peak
        for i in range(local_peak + 1, len(local_env)):
            if local_env[i] < threshold:
                offset = i
                break

        onsets.append(start + onset)
        offsets.append(start + offset)

    return np.array(onsets), np.array(offsets)


def generate_pseudo_labels(
    pcg: np.ndarray,
    sr: int = 2000,
    output_frames: int = 224,
    method: str = "envelope",
    include_s3_s4: bool = False,
) -> np.ndarray:
    """
    Generate pseudo-labels for heart sound segmentation.

    Creates frame-level labels for:
    - Background (0)
    - S1 (1)
    - Systole (2)
    - S2 (3)
    - Diastole (4)
    - S3 (5) - optional
    - S4 (6) - optional

    Args:
        pcg: Input PCG signal (preprocessed)
        sr: Sample rate
        output_frames: Number of output frames (to match spectrogram)
        method: Detection method for heart sounds
        include_s3_s4: Whether to attempt S3/S4 detection

    Returns:
        Label array of shape (output_frames,) with class indices
    """
    # Detect heart sounds
    detection = detect_heart_sounds(pcg, sr, method=method)

    # Initialize labels as background
    n_samples = len(pcg)
    labels = np.zeros(n_samples, dtype=np.int64)

    s1_onsets = detection["s1_onsets"]
    s1_offsets = detection["s1_offsets"]
    s2_onsets = detection["s2_onsets"]
    s2_offsets = detection["s2_offsets"]

    # Label S1 regions
    for onset, offset in zip(s1_onsets, s1_offsets):
        labels[onset:offset] = SEGMENT_CLASSES["S1"]

    # Label S2 regions
    for onset, offset in zip(s2_onsets, s2_offsets):
        labels[onset:offset] = SEGMENT_CLASSES["S2"]

    # Label systole (between S1 offset and S2 onset)
    for i in range(min(len(s1_offsets), len(s2_onsets))):
        if s1_offsets[i] < s2_onsets[i]:
            labels[s1_offsets[i]:s2_onsets[i]] = SEGMENT_CLASSES["systole"]

    # Label diastole (between S2 offset and next S1 onset)
    for i in range(len(s2_offsets)):
        if i + 1 < len(s1_onsets):
            if s2_offsets[i] < s1_onsets[i + 1]:
                labels[s2_offsets[i]:s1_onsets[i + 1]] = SEGMENT_CLASSES["diastole"]
        else:
            # Last diastole extends to end
            labels[s2_offsets[i]:] = SEGMENT_CLASSES["diastole"]

    # Resample to output_frames
    if output_frames != n_samples:
        # Use nearest-neighbor interpolation for labels
        indices = np.linspace(0, n_samples - 1, output_frames).astype(int)
        labels = labels[indices]

    return labels


def validate_pseudo_labels(
    labels: np.ndarray,
    min_s1_count: int = 1,
    min_s2_count: int = 1,
) -> bool:
    """
    Validate pseudo-labels for basic physiological constraints.

    Args:
        labels: Label array
        min_s1_count: Minimum number of S1 segments expected
        min_s2_count: Minimum number of S2 segments expected

    Returns:
        True if labels pass validation
    """
    # Count transitions to detect segments
    s1_mask = labels == SEGMENT_CLASSES["S1"]
    s2_mask = labels == SEGMENT_CLASSES["S2"]

    # Count S1 segments (transitions from non-S1 to S1)
    s1_starts = np.diff(s1_mask.astype(int)) == 1
    s1_count = np.sum(s1_starts)

    # Count S2 segments
    s2_starts = np.diff(s2_mask.astype(int)) == 1
    s2_count = np.sum(s2_starts)

    # Basic validation
    if s1_count < min_s1_count or s2_count < min_s2_count:
        return False

    # Check for reasonable ratio
    if s1_count > 0 and s2_count > 0:
        ratio = s1_count / s2_count
        if ratio < 0.5 or ratio > 2.0:
            return False

    return True
