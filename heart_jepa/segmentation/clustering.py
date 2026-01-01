"""Clustering for separating S1 and S2 heart sounds."""

import torch
import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from .sound_detector import SoundEvent


class HeartSoundClusterer:
    """
    Cluster sound events into S1 and S2 categories.

    Uses K-means or other clustering on frame-level features
    within detected sound regions.
    """

    def __init__(
        self,
        method: str = "kmeans",
        n_clusters: int = 2,
        use_event_features: bool = True,
        feature_aggregation: str = "mean",
    ):
        """
        Initialize clusterer.

        Args:
            method: Clustering method ("kmeans", "gmm", "spectral")
            n_clusters: Number of clusters (default 2 for S1/S2)
            use_event_features: Whether to cluster event features (vs frames)
            feature_aggregation: How to aggregate frame features per event
                                ("mean", "max", "median")
        """
        self.method = method
        self.n_clusters = n_clusters
        self.use_event_features = use_event_features
        self.feature_aggregation = feature_aggregation

    def cluster_events(
        self,
        features: torch.Tensor,
        events: List[SoundEvent],
    ) -> List[int]:
        """
        Cluster sound events based on their features.

        Args:
            features: Frame features of shape (T, D)
            events: List of SoundEvent objects

        Returns:
            List of cluster assignments (0 or 1) for each event
        """
        if len(events) < 2:
            # Not enough events to cluster
            return [0] * len(events)

        # Extract features for each event
        event_features = self._extract_event_features(features, events)

        if event_features.shape[0] < self.n_clusters:
            # Not enough events for clustering
            return list(range(len(events)))

        # Cluster
        cluster_labels = self._cluster(event_features)

        return cluster_labels

    def cluster_batch(
        self,
        features: torch.Tensor,
        events_batch: List[List[SoundEvent]],
    ) -> List[List[int]]:
        """
        Cluster events for a batch of samples.

        Args:
            features: Frame features of shape (B, T, D)
            events_batch: List of event lists for each sample

        Returns:
            List of cluster assignment lists
        """
        B = features.shape[0]
        all_labels = []

        for b in range(B):
            sample_features = features[b]  # (T, D)
            sample_events = events_batch[b]
            labels = self.cluster_events(sample_features, sample_events)
            all_labels.append(labels)

        return all_labels

    def _extract_event_features(
        self,
        features: torch.Tensor,
        events: List[SoundEvent],
    ) -> np.ndarray:
        """Extract aggregated features for each event."""
        features_np = features.cpu().numpy() if torch.is_tensor(features) else features
        T, D = features_np.shape

        event_features = []
        for event in events:
            start = max(0, event.start)
            end = min(T, event.end)

            if end <= start:
                # Empty event, use zeros
                event_feat = np.zeros(D)
            else:
                frame_feats = features_np[start:end]

                if self.feature_aggregation == "mean":
                    event_feat = frame_feats.mean(axis=0)
                elif self.feature_aggregation == "max":
                    event_feat = frame_feats.max(axis=0)
                elif self.feature_aggregation == "median":
                    event_feat = np.median(frame_feats, axis=0)
                else:
                    event_feat = frame_feats.mean(axis=0)

            event_features.append(event_feat)

        return np.array(event_features)

    def _cluster(self, features: np.ndarray) -> List[int]:
        """Apply clustering algorithm."""
        if self.method == "kmeans":
            clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
            )
            labels = clusterer.fit_predict(features)

        elif self.method == "gmm":
            clusterer = GaussianMixture(
                n_components=self.n_clusters,
                random_state=42,
            )
            labels = clusterer.fit_predict(features)

        elif self.method == "spectral":
            # Spectral clustering needs more samples than clusters
            if features.shape[0] <= self.n_clusters:
                return list(range(features.shape[0]))
            clusterer = SpectralClustering(
                n_clusters=self.n_clusters,
                random_state=42,
                affinity='nearest_neighbors',
                n_neighbors=min(5, features.shape[0] - 1),
            )
            labels = clusterer.fit_predict(features)

        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        return labels.tolist()


def assign_s1_s2_labels(
    events: List[SoundEvent],
    cluster_labels: List[int],
    features: Optional[torch.Tensor] = None,
) -> Tuple[List[int], List[int]]:
    """
    Assign S1/S2 labels to clustered events based on temporal ordering.

    In a cardiac cycle:
    - S1 occurs first (start of systole)
    - S2 occurs second (end of systole)
    - Interval S1-S2 (systole) < interval S2-S1 (diastole)

    Args:
        events: List of SoundEvent objects
        cluster_labels: Cluster assignment for each event
        features: Optional features for additional analysis

    Returns:
        Tuple of:
        - s1_indices: Indices of S1 events
        - s2_indices: Indices of S2 events
    """
    if len(events) < 2:
        return [], []

    # Group events by cluster
    cluster_0_events = [(i, e) for i, (e, c) in enumerate(zip(events, cluster_labels)) if c == 0]
    cluster_1_events = [(i, e) for i, (e, c) in enumerate(zip(events, cluster_labels)) if c == 1]

    if not cluster_0_events or not cluster_1_events:
        # All events in same cluster, use alternating assignment
        return _alternating_assignment(events)

    # Determine which cluster is S1 based on timing patterns
    # Strategy: Check which cluster's events tend to come first in pairs

    # Find consecutive event pairs from different clusters
    s1_cluster = _determine_s1_cluster(events, cluster_labels)

    # Assign based on determined S1 cluster
    s1_indices = [i for i, c in enumerate(cluster_labels) if c == s1_cluster]
    s2_indices = [i for i, c in enumerate(cluster_labels) if c != s1_cluster]

    return s1_indices, s2_indices


def _determine_s1_cluster(
    events: List[SoundEvent],
    cluster_labels: List[int],
) -> int:
    """
    Determine which cluster corresponds to S1.

    Uses the timing pattern: S1-S2 interval (systole) is shorter than
    S2-S1 interval (diastole).
    """
    # Compute intervals between consecutive events
    intervals = []
    for i in range(len(events) - 1):
        interval = events[i + 1].start - events[i].end
        from_cluster = cluster_labels[i]
        to_cluster = cluster_labels[i + 1]
        intervals.append((from_cluster, to_cluster, interval))

    # Count average interval for each transition type
    transition_intervals = {}
    for from_c, to_c, interval in intervals:
        key = (from_c, to_c)
        if key not in transition_intervals:
            transition_intervals[key] = []
        transition_intervals[key].append(interval)

    # Average intervals
    avg_intervals = {k: np.mean(v) for k, v in transition_intervals.items()}

    # S1 is followed by short interval (systole)
    # So cluster where (cluster -> other) has shorter average interval is S1
    if len(avg_intervals) < 2:
        # Not enough transitions, default to cluster 0
        return 0

    # Compare (0->1) vs (1->0) intervals
    interval_0_to_1 = avg_intervals.get((0, 1), float('inf'))
    interval_1_to_0 = avg_intervals.get((1, 0), float('inf'))

    # Shorter outgoing interval indicates S1 (followed by systole)
    if interval_0_to_1 < interval_1_to_0:
        return 0
    else:
        return 1


def _alternating_assignment(events: List[SoundEvent]) -> Tuple[List[int], List[int]]:
    """Assign S1/S2 by alternating pattern when clustering fails."""
    s1_indices = []
    s2_indices = []

    for i in range(len(events)):
        if i % 2 == 0:
            s1_indices.append(i)
        else:
            s2_indices.append(i)

    return s1_indices, s2_indices


def cluster_and_label_events(
    features: torch.Tensor,
    events: List[SoundEvent],
    method: str = "kmeans",
) -> Tuple[List[int], List[int]]:
    """
    Convenience function to cluster events and assign S1/S2 labels.

    Args:
        features: Frame features of shape (T, D)
        events: List of SoundEvent objects
        method: Clustering method

    Returns:
        Tuple of (s1_indices, s2_indices)
    """
    clusterer = HeartSoundClusterer(method=method)
    cluster_labels = clusterer.cluster_events(features, events)
    s1_indices, s2_indices = assign_s1_s2_labels(events, cluster_labels, features)

    return s1_indices, s2_indices
