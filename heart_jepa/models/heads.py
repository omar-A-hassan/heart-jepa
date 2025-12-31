"""Task-specific heads for Heart-JEPA."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Segmentation head for heart sound segmentation.

    Takes patch tokens from ViT and produces per-frame segmentation.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 7,
        output_frames: int = 224,
        num_patches: int = 196,
    ):
        """
        Args:
            embed_dim: Input embedding dimension from ViT
            hidden_dim: Hidden dimension for conv layers
            num_classes: Number of segmentation classes
                (S1, Systole, S2, Diastole, S3, S4, Background)
            output_frames: Number of output time frames
            num_patches: Number of input patches (14x14 = 196 for ViT-B/16)
        """
        super().__init__()

        self.num_patches = num_patches
        self.output_frames = output_frames
        self.num_classes = num_classes

        # Project patch tokens
        self.proj = nn.Linear(embed_dim, hidden_dim)

        # 1D conv to process temporal sequence
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # Upsample to target frames
        self.upsample = nn.Upsample(size=output_frames, mode='linear', align_corners=False)

        # Output projection
        self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            patch_tokens: Patch tokens from ViT of shape (B, N, D)
                where N = num_patches (196 for 14x14)

        Returns:
            Segmentation logits of shape (B, num_classes, output_frames)
        """
        B, N, D = patch_tokens.shape

        # Project to hidden dim
        x = self.proj(patch_tokens)  # (B, N, hidden_dim)
        x = self.norm(x)

        # Reshape for 1D conv: (B, hidden_dim, N)
        x = x.transpose(1, 2)

        # Conv layers
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        # Upsample to output frames
        x = self.upsample(x)  # (B, hidden_dim, output_frames)

        # Final classification
        logits = self.head(x)  # (B, num_classes, output_frames)

        return logits


class ClassificationHead(nn.Module):
    """
    Classification head for heart sound diagnosis.

    Takes CLS token and produces classification logits.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            hidden_dim: Hidden dimension for MLP
            num_classes: Number of output classes
                (Normal, Abnormal, Murmur)
            dropout: Dropout probability
        """
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            cls_token: CLS token from ViT of shape (B, D)

        Returns:
            Classification logits of shape (B, num_classes)
        """
        x = self.norm(cls_token)
        return self.mlp(x)


class LinearProbe(nn.Module):
    """
    Simple linear probe for evaluation.

    Used during pre-training to monitor representation quality.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            num_classes: Number of output classes
        """
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Features of shape (B, D)

        Returns:
            Logits of shape (B, num_classes)
        """
        return self.linear(self.norm(x))
