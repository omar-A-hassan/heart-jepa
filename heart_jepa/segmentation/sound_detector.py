"""Sound event detection using attention-based saliency."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SoundEvent:
    """Represents a detected sound event."""
    start: int  # Start frame index
    end: int  # End frame index (exclusive)
    peak: int  # Peak frame index
    intensity: float  # Peak intensity/saliency

    @property
    def duration(self) -> int:
        """Duration in frames."""
        return self.end - self.start

    @property
    def center(self) -> float:
        """Center position."""
        return (self.start + self.end) / 2


class SoundDetector:
    """
    Detect sound events from attention-based saliency.

    Uses thresholding and morphological operations to find
    sound regions in the temporal saliency signal.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        min_duration: int = 5,
        max_gap: int = 3,
        smooth_window: int = 5,
    ):
        """
        Initialize sound detector.

        Args:
            threshold: Saliency threshold for sound detection (0-1)
            min_duration: Minimum duration of sound event in frames
            max_gap: Maximum gap to merge adjacent events
            smooth_window: Window size for smoothing saliency
        """
        self.threshold = threshold
        self.min_duration = min_duration
        self.max_gap = max_gap
        self.smooth_window = smooth_window

    def detect(
        self,
        saliency: torch.Tensor,
        adaptive_threshold: bool = True,
    ) -> Tuple[torch.Tensor, List[List[SoundEvent]]]:
        """
        Detect sound events from saliency signal.

        Args:
            saliency: Temporal saliency of shape (B, T)
            adaptive_threshold: Use adaptive (per-sample) thresholding

        Returns:
            Tuple of:
            - sound_mask: Binary mask of shape (B, T)
            - events: List of SoundEvent lists for each sample
        """
        B, T = saliency.shape

        # Smooth saliency
        if self.smooth_window > 1:
            saliency = self._smooth(saliency)

        # Normalize to 0-1
        saliency = self._normalize(saliency)

        # Threshold
        if adaptive_threshold:
            threshold = self._compute_adaptive_threshold(saliency)
        else:
            threshold = self.threshold

        # Create binary mask
        if isinstance(threshold, torch.Tensor):
            sound_mask = saliency > threshold.unsqueeze(1)
        else:
            sound_mask = saliency > threshold

        # Post-process mask
        sound_mask = self._postprocess_mask(sound_mask)

        # Extract events
        events = self._extract_events(sound_mask, saliency)

        return sound_mask, events

    def _smooth(self, saliency: torch.Tensor) -> torch.Tensor:
        """Apply smoothing to saliency signal."""
        kernel = torch.ones(1, 1, self.smooth_window, device=saliency.device)
        kernel = kernel / self.smooth_window

        # Add channel dim for conv1d
        saliency = saliency.unsqueeze(1)  # (B, 1, T)
        padding = self.smooth_window // 2
        saliency = F.conv1d(saliency, kernel, padding=padding)
        saliency = saliency.squeeze(1)  # (B, T)

        return saliency

    def _normalize(self, saliency: torch.Tensor) -> torch.Tensor:
        """Normalize saliency to 0-1 range per sample."""
        # Per-sample min-max normalization
        min_val = saliency.min(dim=1, keepdim=True).values
        max_val = saliency.max(dim=1, keepdim=True).values
        range_val = max_val - min_val + 1e-8

        normalized = (saliency - min_val) / range_val
        return normalized

    def _compute_adaptive_threshold(
        self,
        saliency: torch.Tensor,
    ) -> torch.Tensor:
        """Compute adaptive threshold based on signal statistics."""
        # Use Otsu-like approach: find threshold that maximizes between-class variance
        # Simplified: use mean + 0.5 * std
        mean = saliency.mean(dim=1)
        std = saliency.std(dim=1)
        threshold = mean + 0.5 * std

        # Clamp to reasonable range
        threshold = torch.clamp(threshold, 0.2, 0.8)

        return threshold

    def _postprocess_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply morphological operations to clean up mask."""
        # Close small gaps (dilation then erosion)
        if self.max_gap > 0:
            mask = self._dilate(mask, self.max_gap)
            mask = self._erode(mask, self.max_gap)

        # Remove small segments (erosion then dilation)
        if self.min_duration > 1:
            erosion_size = self.min_duration // 2
            mask = self._erode(mask, erosion_size)
            mask = self._dilate(mask, erosion_size)

        return mask

    def _dilate(self, mask: torch.Tensor, size: int) -> torch.Tensor:
        """Binary dilation using max pooling."""
        if size <= 0:
            return mask
        mask = mask.float().unsqueeze(1)  # (B, 1, T)
        mask = F.max_pool1d(mask, kernel_size=2*size+1, stride=1, padding=size)
        mask = mask.squeeze(1) > 0.5
        return mask

    def _erode(self, mask: torch.Tensor, size: int) -> torch.Tensor:
        """Binary erosion using min pooling (inverted max pooling)."""
        if size <= 0:
            return mask
        mask = mask.float().unsqueeze(1)  # (B, 1, T)
        # Min pooling = invert, max pool, invert
        mask = 1 - mask
        mask = F.max_pool1d(mask, kernel_size=2*size+1, stride=1, padding=size)
        mask = 1 - mask
        mask = mask.squeeze(1) > 0.5
        return mask

    def _extract_events(
        self,
        mask: torch.Tensor,
        saliency: torch.Tensor,
    ) -> List[List[SoundEvent]]:
        """Extract SoundEvent objects from binary mask."""
        B, T = mask.shape
        all_events = []

        mask_np = mask.cpu().numpy()
        saliency_np = saliency.cpu().numpy()

        for b in range(B):
            events = []
            sample_mask = mask_np[b]
            sample_saliency = saliency_np[b]

            # Find connected components (runs of True)
            in_event = False
            start = 0

            for t in range(T):
                if sample_mask[t] and not in_event:
                    # Start of event
                    in_event = True
                    start = t
                elif not sample_mask[t] and in_event:
                    # End of event
                    in_event = False
                    end = t

                    if end - start >= self.min_duration:
                        # Find peak within event
                        event_saliency = sample_saliency[start:end]
                        peak_local = int(np.argmax(event_saliency))
                        peak = start + peak_local
                        intensity = float(event_saliency[peak_local])

                        events.append(SoundEvent(
                            start=start,
                            end=end,
                            peak=peak,
                            intensity=intensity,
                        ))

            # Handle event at end of signal
            if in_event and T - start >= self.min_duration:
                event_saliency = sample_saliency[start:T]
                peak_local = int(np.argmax(event_saliency))
                peak = start + peak_local
                intensity = float(event_saliency[peak_local])

                events.append(SoundEvent(
                    start=start,
                    end=T,
                    peak=peak,
                    intensity=intensity,
                ))

            all_events.append(events)

        return all_events


def saliency_to_sound_mask(
    saliency: torch.Tensor,
    threshold: float = 0.3,
    min_duration: int = 5,
) -> torch.Tensor:
    """
    Simple function to convert saliency to binary sound mask.

    Args:
        saliency: Temporal saliency of shape (B, T)
        threshold: Threshold for detection
        min_duration: Minimum event duration

    Returns:
        Binary mask of shape (B, T)
    """
    detector = SoundDetector(
        threshold=threshold,
        min_duration=min_duration,
    )
    mask, _ = detector.detect(saliency, adaptive_threshold=False)
    return mask


def find_sound_events(
    saliency: torch.Tensor,
    **kwargs,
) -> List[List[SoundEvent]]:
    """
    Find sound events in saliency signal.

    Args:
        saliency: Temporal saliency of shape (B, T)
        **kwargs: Arguments passed to SoundDetector

    Returns:
        List of SoundEvent lists for each sample
    """
    detector = SoundDetector(**kwargs)
    _, events = detector.detect(saliency)
    return events
