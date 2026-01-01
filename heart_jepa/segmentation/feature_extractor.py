"""Feature extraction for self-supervised segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FeatureExtractor:
    """
    Extract frame-level features from ViT encoder for clustering.

    Converts patch tokens to temporal frame features.
    """

    def __init__(
        self,
        model: nn.Module,
        output_frames: int = 224,
        normalize: bool = True,
    ):
        """
        Initialize feature extractor.

        Args:
            model: ViT model or HeartJEPA model
            output_frames: Number of output temporal frames
            normalize: Whether to L2-normalize features
        """
        self.model = model
        self.output_frames = output_frames
        self.normalize = normalize

        # Find embed_dim
        self.embed_dim = self._get_embed_dim()

    def _get_embed_dim(self) -> int:
        """Get embedding dimension from model."""
        if hasattr(self.model, 'embed_dim'):
            return self.model.embed_dim

        if hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
            if hasattr(encoder, 'embed_dim'):
                return encoder.embed_dim
            if hasattr(encoder, 'encoder'):
                inner = encoder.encoder
                if hasattr(inner, 'embed_dim'):
                    return inner.embed_dim

        # Default for ViT-Base
        return 768

    def extract(
        self,
        x: torch.Tensor,
        return_cls: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract frame-level features.

        Args:
            x: Input spectrogram of shape (B, C, H, W)
            return_cls: Whether to also return CLS token

        Returns:
            Tuple of:
            - frame_features: (B, output_frames, embed_dim)
            - cls_token: (B, embed_dim) if return_cls else None
        """
        features = extract_patch_features(self.model, x)
        # features: (B, N+1, embed_dim) where N = num_patches

        # Separate CLS and patch tokens
        cls_token = features[:, 0]  # (B, embed_dim)
        patch_tokens = features[:, 1:]  # (B, N, embed_dim)

        # Convert patches to temporal frames
        frame_features = patches_to_temporal(
            patch_tokens,
            output_frames=self.output_frames,
        )  # (B, output_frames, embed_dim)

        # Normalize if requested
        if self.normalize:
            frame_features = F.normalize(frame_features, p=2, dim=-1)

        if return_cls:
            return frame_features, cls_token
        return frame_features, None


def extract_patch_features(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Extract all token features (CLS + patches) from model.

    Args:
        model: ViT model or HeartJEPA model
        x: Input tensor of shape (B, C, H, W)

    Returns:
        Features of shape (B, N+1, embed_dim)
    """
    with torch.no_grad():
        # Try different model structures
        if hasattr(model, 'encoder'):
            # HeartJEPA model
            encoder = model.encoder
            if hasattr(encoder, 'encoder'):
                # HeartJEPAEncoder -> ViTEncoder
                inner = encoder.encoder
                if hasattr(inner, 'forward_features'):
                    features = inner.forward_features(x)
                elif hasattr(inner, 'backbone'):
                    features = inner.backbone.forward_features(x)
                else:
                    features = inner(x)
            elif hasattr(encoder, 'forward_features'):
                features = encoder.forward_features(x)
            else:
                features = encoder(x)
        elif hasattr(model, 'forward_features'):
            features = model.forward_features(x)
        else:
            # Assume direct ViT
            features = model(x)

        # Ensure we have the right shape
        if features.dim() == 2:
            # Only CLS token returned, need full features
            raise ValueError(
                "Model only returned CLS token. Use forward_features method."
            )

        return features


def patches_to_temporal(
    patch_tokens: torch.Tensor,
    output_frames: int,
    grid_mode: str = "average_freq",
) -> torch.Tensor:
    """
    Convert patch tokens to temporal frame features.

    For spectrograms, patches are arranged in a 2D grid where:
    - Height axis = frequency
    - Width axis = time

    Args:
        patch_tokens: Patch tokens of shape (B, N, D) where N = H*W patches
        output_frames: Number of output temporal frames
        grid_mode: How to handle 2D grid
            "average_freq" = average across frequency (height)
            "flatten" = treat all patches as temporal sequence
            "concat_freq" = concatenate frequency bands

    Returns:
        Temporal features of shape (B, output_frames, D) or (B, output_frames, D*H)
    """
    B, N, D = patch_tokens.shape

    # Determine grid size (assuming square)
    grid_size = int(N ** 0.5)
    if grid_size * grid_size != N:
        # Non-square patches, treat as 1D sequence
        grid_mode = "flatten"
        temporal_len = N
    else:
        temporal_len = grid_size

    if grid_mode == "flatten":
        # Treat all patches as temporal sequence
        temporal_features = patch_tokens  # (B, N, D)
    elif grid_mode == "average_freq":
        # Reshape to 2D grid and average across frequency
        patch_tokens = patch_tokens.view(B, grid_size, grid_size, D)
        temporal_features = patch_tokens.mean(dim=1)  # (B, grid_size, D)
    elif grid_mode == "concat_freq":
        # Reshape and concatenate frequency bands
        patch_tokens = patch_tokens.view(B, grid_size, grid_size, D)
        temporal_features = patch_tokens.permute(0, 2, 1, 3)  # (B, W, H, D)
        temporal_features = temporal_features.reshape(B, grid_size, -1)  # (B, W, H*D)
    else:
        raise ValueError(f"Unknown grid_mode: {grid_mode}")

    # Interpolate to output_frames
    # Shape: (B, T, D) -> transpose -> (B, D, T)
    temporal_features = temporal_features.transpose(1, 2)
    temporal_features = F.interpolate(
        temporal_features,
        size=output_frames,
        mode='linear',
        align_corners=False,
    )
    # Shape: (B, D, output_frames) -> transpose -> (B, output_frames, D)
    temporal_features = temporal_features.transpose(1, 2)

    return temporal_features


def get_frame_similarities(
    features: torch.Tensor,
    method: str = "cosine",
) -> torch.Tensor:
    """
    Compute pairwise similarities between frames.

    Useful for boundary detection and clustering.

    Args:
        features: Frame features of shape (B, T, D)
        method: Similarity method ("cosine", "euclidean", "dot")

    Returns:
        Similarity matrix of shape (B, T, T)
    """
    if method == "cosine":
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)
        similarity = torch.bmm(features_norm, features_norm.transpose(1, 2))
    elif method == "dot":
        similarity = torch.bmm(features, features.transpose(1, 2))
    elif method == "euclidean":
        # Pairwise euclidean distance (converted to similarity)
        B, T, D = features.shape
        diff = features.unsqueeze(2) - features.unsqueeze(1)  # (B, T, T, D)
        dist = torch.norm(diff, p=2, dim=-1)  # (B, T, T)
        similarity = 1.0 / (1.0 + dist)  # Convert to similarity
    else:
        raise ValueError(f"Unknown similarity method: {method}")

    return similarity


def compute_feature_gradients(
    features: torch.Tensor,
) -> torch.Tensor:
    """
    Compute temporal gradients of features.

    High gradients indicate potential segment boundaries.

    Args:
        features: Frame features of shape (B, T, D)

    Returns:
        Gradient magnitudes of shape (B, T-1)
    """
    # Compute difference between consecutive frames
    diff = features[:, 1:] - features[:, :-1]  # (B, T-1, D)

    # Compute magnitude of difference
    gradient = torch.norm(diff, p=2, dim=-1)  # (B, T-1)

    return gradient
