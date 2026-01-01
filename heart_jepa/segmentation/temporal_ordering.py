"""Temporal ordering for cardiac phase assignment."""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .sound_detector import SoundEvent


# Segment class indices (matching pseudo_labels.py)
SEGMENT_CLASSES = {
    "background": 0,
    "S1": 1,
    "systole": 2,
    "S2": 3,
    "diastole": 4,
    "S3": 5,
    "S4": 6,
}


@dataclass
class CardiacCycle:
    """Represents a single cardiac cycle."""
    s1_event: Optional[SoundEvent]
    s2_event: Optional[SoundEvent]
    cycle_start: int  # Frame index
    cycle_end: int  # Frame index

    @property
    def systole_start(self) -> Optional[int]:
        """Start of systole (end of S1)."""
        return self.s1_event.end if self.s1_event else None

    @property
    def systole_end(self) -> Optional[int]:
        """End of systole (start of S2)."""
        return self.s2_event.start if self.s2_event else None

    @property
    def diastole_start(self) -> Optional[int]:
        """Start of diastole (end of S2)."""
        return self.s2_event.end if self.s2_event else None

    @property
    def diastole_end(self) -> int:
        """End of diastole (end of cycle)."""
        return self.cycle_end


class TemporalOrderer:
    """
    Assign cardiac phase labels based on temporal ordering.

    Uses physiological constraints:
    - Heart rate: 40-200 bpm (0.3-1.5s per beat)
    - Systole is ~35% of cardiac cycle
    - S1-S2 interval < S2-S1 interval
    """

    def __init__(
        self,
        min_hr: float = 40,
        max_hr: float = 200,
        systole_ratio: float = 0.35,
        frame_rate: float = 44.8,  # frames per second (224 frames / 5s)
    ):
        """
        Initialize temporal orderer.

        Args:
            min_hr: Minimum heart rate in bpm
            max_hr: Maximum heart rate in bpm
            systole_ratio: Expected systole / cycle duration ratio
            frame_rate: Number of frames per second
        """
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.systole_ratio = systole_ratio
        self.frame_rate = frame_rate

        # Compute frame limits
        self.min_cycle_frames = int(60 / max_hr * frame_rate)  # ~13 frames at 200bpm
        self.max_cycle_frames = int(60 / min_hr * frame_rate)  # ~67 frames at 40bpm

    def order_events(
        self,
        events: List[SoundEvent],
        s1_indices: List[int],
        s2_indices: List[int],
    ) -> List[CardiacCycle]:
        """
        Organize events into cardiac cycles.

        Args:
            events: All detected sound events
            s1_indices: Indices of S1 events
            s2_indices: Indices of S2 events

        Returns:
            List of CardiacCycle objects
        """
        if not events:
            return []

        # Get S1 and S2 events
        s1_events = [events[i] for i in s1_indices if i < len(events)]
        s2_events = [events[i] for i in s2_indices if i < len(events)]

        # Sort by time
        s1_events = sorted(s1_events, key=lambda e: e.start)
        s2_events = sorted(s2_events, key=lambda e: e.start)

        # Match S1 and S2 into cycles
        cycles = self._match_cycles(s1_events, s2_events)

        return cycles

    def _match_cycles(
        self,
        s1_events: List[SoundEvent],
        s2_events: List[SoundEvent],
    ) -> List[CardiacCycle]:
        """Match S1 and S2 events into cardiac cycles."""
        cycles = []

        # Use greedy matching: for each S1, find the next S2
        s2_idx = 0
        for i, s1 in enumerate(s1_events):
            # Find next S2 after this S1
            while s2_idx < len(s2_events) and s2_events[s2_idx].start < s1.end:
                s2_idx += 1

            if s2_idx < len(s2_events):
                s2 = s2_events[s2_idx]

                # Check if interval is physiologically plausible
                interval = s2.start - s1.end
                if interval > 0 and interval < self.max_cycle_frames * self.systole_ratio * 2:
                    # Valid S1-S2 pair
                    # Cycle ends at next S1 or end of signal
                    if i + 1 < len(s1_events):
                        cycle_end = s1_events[i + 1].start
                    else:
                        # Estimate cycle end based on typical diastole duration
                        expected_diastole = int(
                            (s2.end - s1.start) * (1 - self.systole_ratio) / self.systole_ratio
                        )
                        cycle_end = s2.end + expected_diastole

                    cycles.append(CardiacCycle(
                        s1_event=s1,
                        s2_event=s2,
                        cycle_start=s1.start,
                        cycle_end=cycle_end,
                    ))

                    s2_idx += 1

        return cycles

    def generate_frame_labels(
        self,
        cycles: List[CardiacCycle],
        total_frames: int,
    ) -> np.ndarray:
        """
        Generate frame-level segmentation labels from cycles.

        Args:
            cycles: List of cardiac cycles
            total_frames: Total number of frames

        Returns:
            Label array of shape (total_frames,)
        """
        labels = np.zeros(total_frames, dtype=np.int64)  # Background

        for cycle in cycles:
            # Label S1
            if cycle.s1_event:
                start = max(0, cycle.s1_event.start)
                end = min(total_frames, cycle.s1_event.end)
                labels[start:end] = SEGMENT_CLASSES["S1"]

            # Label systole (between S1 and S2)
            if cycle.systole_start and cycle.systole_end:
                start = max(0, cycle.systole_start)
                end = min(total_frames, cycle.systole_end)
                if end > start:
                    labels[start:end] = SEGMENT_CLASSES["systole"]

            # Label S2
            if cycle.s2_event:
                start = max(0, cycle.s2_event.start)
                end = min(total_frames, cycle.s2_event.end)
                labels[start:end] = SEGMENT_CLASSES["S2"]

            # Label diastole (after S2 until cycle end)
            if cycle.diastole_start:
                start = max(0, cycle.diastole_start)
                end = min(total_frames, cycle.diastole_end)
                if end > start:
                    labels[start:end] = SEGMENT_CLASSES["diastole"]

        return labels


def assign_labels_by_intervals(
    events: List[SoundEvent],
    total_frames: int,
) -> np.ndarray:
    """
    Assign S1/S2/Systole/Diastole labels using interval-based ordering.

    Algorithm:
    1. Find pairs of consecutive events
    2. Shorter interval = systole (S1-S2), longer = diastole (S2-S1)
    3. First sound after long interval = S1
    4. Second sound after short interval = S2

    Args:
        events: List of sound events (sorted by time)
        total_frames: Total number of output frames

    Returns:
        Label array of shape (total_frames,)
    """
    if len(events) < 2:
        return _label_single_event(events, total_frames)

    # Sort events by start time
    events = sorted(events, key=lambda e: e.start)

    # Compute intervals between consecutive events
    intervals = []
    for i in range(len(events) - 1):
        gap = events[i + 1].start - events[i].end
        intervals.append(gap)

    # Classify intervals as systole (short) or diastole (long)
    # Use median to distinguish
    median_interval = np.median(intervals)

    # Assign labels
    labels = np.zeros(total_frames, dtype=np.int64)

    # Determine first event type
    # If first interval is short, first event is S1 (followed by systole)
    # If first interval is long, first event is S2 (followed by diastole)
    if intervals[0] < median_interval:
        first_is_s1 = True
    else:
        first_is_s1 = False

    # Label each event and fill phases
    for i, event in enumerate(events):
        is_s1 = (i % 2 == 0) == first_is_s1

        start = max(0, event.start)
        end = min(total_frames, event.end)

        if is_s1:
            labels[start:end] = SEGMENT_CLASSES["S1"]

            # Fill systole until next event
            if i + 1 < len(events):
                phase_end = min(total_frames, events[i + 1].start)
                if phase_end > end:
                    labels[end:phase_end] = SEGMENT_CLASSES["systole"]
        else:
            labels[start:end] = SEGMENT_CLASSES["S2"]

            # Fill diastole until next event
            if i + 1 < len(events):
                phase_end = min(total_frames, events[i + 1].start)
                if phase_end > end:
                    labels[end:phase_end] = SEGMENT_CLASSES["diastole"]
            else:
                # Last event, fill diastole to end
                labels[end:total_frames] = SEGMENT_CLASSES["diastole"]

    return labels


def _label_single_event(
    events: List[SoundEvent],
    total_frames: int,
) -> np.ndarray:
    """Handle case with 0 or 1 events."""
    labels = np.zeros(total_frames, dtype=np.int64)

    if events:
        event = events[0]
        start = max(0, event.start)
        end = min(total_frames, event.end)
        # Default to S1 for single event
        labels[start:end] = SEGMENT_CLASSES["S1"]

    return labels


def validate_segmentation(
    labels: np.ndarray,
    min_hr: float = 40,
    max_hr: float = 200,
    frame_rate: float = 44.8,
) -> Tuple[bool, str]:
    """
    Validate segmentation labels for physiological plausibility.

    Args:
        labels: Frame-level labels
        min_hr: Minimum heart rate
        max_hr: Maximum heart rate
        frame_rate: Frames per second

    Returns:
        Tuple of (is_valid, message)
    """
    # Count S1 and S2 occurrences
    s1_mask = labels == SEGMENT_CLASSES["S1"]
    s2_mask = labels == SEGMENT_CLASSES["S2"]

    # Count segments (transitions)
    s1_starts = np.diff(s1_mask.astype(int)) == 1
    s2_starts = np.diff(s2_mask.astype(int)) == 1

    s1_count = np.sum(s1_starts) + (1 if s1_mask[0] else 0)
    s2_count = np.sum(s2_starts) + (1 if s2_mask[0] else 0)

    # Check counts
    if s1_count == 0 or s2_count == 0:
        return False, "Missing S1 or S2 segments"

    if abs(s1_count - s2_count) > 1:
        return False, f"Unbalanced S1/S2 counts: {s1_count} vs {s2_count}"

    # Check timing
    duration_seconds = len(labels) / frame_rate
    expected_min_beats = duration_seconds * min_hr / 60
    expected_max_beats = duration_seconds * max_hr / 60

    if s1_count < expected_min_beats * 0.5:
        return False, f"Too few beats detected: {s1_count}"

    if s1_count > expected_max_beats * 1.5:
        return False, f"Too many beats detected: {s1_count}"

    return True, "Valid segmentation"
