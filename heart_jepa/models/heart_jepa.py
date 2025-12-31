"""Main Heart-JEPA model combining encoder with task heads."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .encoder import HeartJEPAEncoder
from .heads import SegmentationHead, ClassificationHead


class HeartJEPA(nn.Module):
    """
    Heart-JEPA: Joint-Embedding Predictive Architecture for PCG analysis.

    Combines:
    - ViT encoder with projector (for LEJEPA pre-training)
    - Segmentation head (S1, S2, S3, S4, systole, diastole, background)
    - Classification head (normal/abnormal/murmur)
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        proj_dim: int = 256,
        hidden_dim: int = 2048,
        seg_hidden_dim: int = 256,
        seg_num_classes: int = 7,
        seg_output_frames: int = 224,
        cls_hidden_dim: int = 256,
        cls_num_classes: int = 3,
        cls_dropout: float = 0.1,
        in_chans: int = 1,
    ):
        """
        Args:
            backbone: timm model name for ViT backbone
            pretrained: Whether to use ImageNet pretrained weights
            proj_dim: Projection dimension for LEJEPA
            hidden_dim: Hidden dimension for projector MLP
            seg_hidden_dim: Hidden dimension for segmentation head
            seg_num_classes: Number of segmentation classes
            seg_output_frames: Output temporal resolution for segmentation
            cls_hidden_dim: Hidden dimension for classification head
            cls_num_classes: Number of classification classes
            cls_dropout: Dropout for classification head
            in_chans: Number of input channels (1 for spectrogram)
        """
        super().__init__()

        # Main encoder with projector
        self.encoder = HeartJEPAEncoder(
            backbone=backbone,
            pretrained=pretrained,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            in_chans=in_chans,
        )

        embed_dim = self.encoder.embed_dim

        # Segmentation head
        self.seg_head = SegmentationHead(
            embed_dim=embed_dim,
            hidden_dim=seg_hidden_dim,
            num_classes=seg_num_classes,
            output_frames=seg_output_frames,
        )

        # Classification head
        self.cls_head = ClassificationHead(
            embed_dim=embed_dim,
            hidden_dim=cls_hidden_dim,
            num_classes=cls_num_classes,
            dropout=cls_dropout,
        )

        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through Heart-JEPA.

        Args:
            x: Input spectrogram of shape (B, C, H, W) or (B, V, C, H, W)
            return_features: Whether to return intermediate features

        Returns:
            Tuple of:
            - proj: Projections for LEJEPA of shape (B, V, proj_dim)
            - seg_logits: Segmentation logits of shape (B, seg_classes, frames)
            - cls_logits: Classification logits of shape (B, cls_classes)
            - (optional) features: All token features if return_features=True
        """
        # Handle multi-view input for pretraining
        if x.dim() == 5:
            B, V, C, H, W = x.shape
            # For downstream tasks, use first view
            x_single = x[:, 0]  # (B, C, H, W)
        else:
            B = x.shape[0]
            V = 1
            x_single = x

        # Get features and projections
        features, proj = self.encoder(x, return_all_tokens=True)
        # features: (B*V, N+1, embed_dim), proj: (B, V, proj_dim)

        # For segmentation/classification, use features from first view
        if x.dim() == 5:
            # Take first view's features
            features_single = features[:B]  # (B, N+1, embed_dim)
        else:
            features_single = features  # (B, N+1, embed_dim)

        # CLS token for classification
        cls_token = features_single[:, 0]  # (B, embed_dim)

        # Patch tokens for segmentation
        patch_tokens = features_single[:, 1:]  # (B, N, embed_dim)

        # Segmentation
        seg_logits = self.seg_head(patch_tokens)  # (B, seg_classes, frames)

        # Classification
        cls_logits = self.cls_head(cls_token)  # (B, cls_classes)

        if return_features:
            return proj, seg_logits, cls_logits, features
        return proj, seg_logits, cls_logits

    def forward_encoder(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder only (for LEJEPA pre-training).

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, V, C, H, W)
            return_all_tokens: Whether to return all patch tokens

        Returns:
            Tuple of (features, projections)
        """
        return self.encoder(x, return_all_tokens=return_all_tokens)

    def forward_segmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for segmentation only.

        Args:
            x: Input spectrogram of shape (B, C, H, W)

        Returns:
            Segmentation logits of shape (B, seg_classes, frames)
        """
        features = self.encoder.encoder.forward_features(x)  # (B, N+1, embed_dim)
        patch_tokens = features[:, 1:]  # (B, N, embed_dim)
        return self.seg_head(patch_tokens)

    def forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification only.

        Args:
            x: Input spectrogram of shape (B, C, H, W)

        Returns:
            Classification logits of shape (B, cls_classes)
        """
        cls_token = self.encoder.encoder(x)  # (B, embed_dim)
        return self.cls_head(cls_token)

    def get_encoder_params(self):
        """Get encoder parameters (for freezing during fine-tuning)."""
        return self.encoder.parameters()

    def get_head_params(self):
        """Get task head parameters (for fine-tuning)."""
        return list(self.seg_head.parameters()) + list(self.cls_head.parameters())

    def freeze_encoder(self):
        """Freeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True
