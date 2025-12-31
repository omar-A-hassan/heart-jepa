"""Segmentation losses for heart sound segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SegmentationLoss(nn.Module):
    """
    Combined loss for heart sound segmentation.

    Combines:
    - Cross-entropy loss for frame classification
    - Dice loss for class-wise overlap
    - Optional focal loss for class imbalance
    """

    def __init__(
        self,
        num_classes: int = 7,
        class_weights: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        focal_gamma: float = 0.0,
        smooth: float = 1e-6,
        ignore_index: int = -100,
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            class_weights: Optional weights for each class (for CE loss)
            dice_weight: Weight for Dice loss
            ce_weight: Weight for cross-entropy loss
            focal_gamma: Gamma for focal loss (0 = no focal weighting)
            smooth: Smoothing factor for Dice loss
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_gamma = focal_gamma
        self.smooth = smooth
        self.ignore_index = ignore_index

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined segmentation loss.

        Args:
            logits: Predicted logits of shape (B, C, T)
            targets: Target labels of shape (B, T) with class indices

        Returns:
            Combined loss value
        """
        B, C, T = logits.shape

        # Cross-entropy loss
        ce_loss = self._compute_ce_loss(logits, targets)

        # Dice loss
        dice_loss = self._compute_dice_loss(logits, targets)

        # Combined loss
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return loss

    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss with optional focal weighting."""
        # Standard cross-entropy
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Apply focal weighting if gamma > 0
        if self.focal_gamma > 0:
            # Get predicted probabilities for true class
            probs = F.softmax(logits, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt) ** self.focal_gamma
            ce_loss = focal_weight * ce_loss

        return ce_loss.mean()

    def _compute_dice_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss for each class."""
        B, C, T = logits.shape

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, T)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=C)  # (B, T, C)
        targets_one_hot = targets_one_hot.permute(0, 2, 1).float()  # (B, C, T)

        # Compute Dice coefficient for each class
        intersection = (probs * targets_one_hot).sum(dim=(0, 2))
        union = probs.sum(dim=(0, 2)) + targets_one_hot.sum(dim=(0, 2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average over classes
        dice_loss = 1.0 - dice.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    Focuses learning on hard examples by down-weighting easy examples.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weights
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits of shape (B, C, ...) or (B, C)
            targets: Target labels

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Get predicted probabilities for true class
        probs = F.softmax(logits, dim=1)
        if logits.dim() == 2:
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            # For segmentation: logits (B, C, T), targets (B, T)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * ce_loss

        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.view(-1))
            alpha_t = alpha_t.view(targets.shape)
            loss = alpha_t * loss

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
