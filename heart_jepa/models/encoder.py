"""Encoder architectures for Heart-JEPA."""

import torch
import torch.nn as nn
import timm


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder for spectrogram feature extraction.

    Uses pretrained ViT from timm library.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        in_chans: int = 1,
        drop_path_rate: float = 0.1,
    ):
        """
        Args:
            model_name: timm model name
            pretrained: Whether to use pretrained weights
            in_chans: Number of input channels (1 for spectrogram)
            drop_path_rate: Drop path rate for regularization
        """
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
        )

        self.embed_dim = self.backbone.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Features of shape (B, embed_dim) for CLS token
            or (B, N, embed_dim) for all tokens
        """
        return self.backbone(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get all token features including CLS.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            All token features of shape (B, N+1, embed_dim)
        """
        return self.backbone.forward_features(x)


class HeartJEPAEncoder(nn.Module):
    """
    Heart-JEPA encoder with projector head.

    Combines ViT backbone with MLP projector for LEJEPA training.
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        proj_dim: int = 256,
        hidden_dim: int = 2048,
        in_chans: int = 1,
    ):
        """
        Args:
            backbone: timm model name
            pretrained: Whether to use pretrained weights
            proj_dim: Output dimension of projector
            hidden_dim: Hidden dimension of projector MLP
            in_chans: Number of input channels
        """
        super().__init__()

        # Backbone
        self.encoder = ViTEncoder(
            model_name=backbone,
            pretrained=pretrained,
            in_chans=in_chans,
        )

        embed_dim = self.encoder.embed_dim

        # Projector MLP
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> tuple:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, V, C, H, W) for multi-view
            return_all_tokens: Whether to return all patch tokens

        Returns:
            Tuple of (embeddings, projections)
            - embeddings: (B*V, embed_dim) or (B*V, N, embed_dim) if return_all_tokens
            - projections: (B, V, proj_dim) for multi-view input
        """
        # Handle multi-view input
        if x.dim() == 5:
            B, V, C, H, W = x.shape
            x = x.flatten(0, 1)  # (B*V, C, H, W)
        else:
            B = x.shape[0]
            V = 1

        # Get features
        if return_all_tokens:
            features = self.encoder.forward_features(x)  # (B*V, N+1, embed_dim)
            cls_token = features[:, 0]  # (B*V, embed_dim)
        else:
            cls_token = self.encoder(x)  # (B*V, embed_dim)
            features = cls_token

        # Project
        proj = self.projector(cls_token)  # (B*V, proj_dim)

        # Reshape for multi-view
        proj = proj.view(B, V, -1)  # (B, V, proj_dim)

        if return_all_tokens:
            return features, proj
        else:
            return cls_token, proj
