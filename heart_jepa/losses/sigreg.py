"""SIGReg loss using official LEJEPA library.

Uses the Epps-Pulley test with SlicingUnivariateTest for multivariate
normality testing as described in the LEJEPA paper.
"""

import torch
import torch.nn as nn

from lejepa.univariate import EppsPulley
from lejepa.multivariate import SlicingUnivariateTest


class SIGReg(nn.Module):
    """
    SIGReg loss module using official LEJEPA implementation.

    Uses:
    - EppsPulley: Univariate normality test via characteristic functions
    - SlicingUnivariateTest: Extends to multivariate via random projections

    Encourages embeddings to follow isotropic Gaussian distribution.
    """

    def __init__(
        self,
        num_slices: int = 1000,
        t_max: float = 3.0,
        n_points: int = 17,
        weight: float = 1.0,
        clip_value: float = None,
    ):
        """
        Args:
            num_slices: Number of random projections (paper uses 1000)
            t_max: Max integration point for Epps-Pulley
            n_points: Number of integration points (must be odd)
            weight: Loss weight multiplier (lambda in paper)
            clip_value: Clip statistics below this value to 0
        """
        super().__init__()
        self.weight = weight

        # Create official LEJEPA test
        univariate_test = EppsPulley(
            t_max=t_max,
            n_points=n_points,
        )

        self.test = SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=num_slices,
            reduction="mean",
            clip_value=clip_value,
        )

    def forward(self, embeddings: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute SIGReg loss.

        Args:
            embeddings: Embedding tensor of shape (B, D) or (B, V, D)
            eps: Small epsilon for numerical stability

        Returns:
            Weighted SIGReg loss
        """
        if embeddings.dim() == 3:
            # Multi-view: flatten batch and views
            B, V, D = embeddings.shape
            embeddings = embeddings.reshape(B * V, D)

        B, D = embeddings.shape
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Standardize embeddings (required for normality test)
        mean = embeddings.mean(dim=0, keepdim=True)
        std = embeddings.std(dim=0, keepdim=True) + eps
        z = (embeddings - mean) / std

        # Compute SIGReg using official LEJEPA test
        loss = self.test(z)
        return self.weight * loss


def sigreg_loss(
    embeddings: torch.Tensor,
    num_slices: int = 1000,
    t_max: float = 3.0,
    n_points: int = 17,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Functional interface for SIGReg loss using official LEJEPA.

    Args:
        embeddings: Embedding tensor of shape (B, D)
        num_slices: Number of random projections
        t_max: Max integration point for Epps-Pulley
        n_points: Number of integration points (must be odd)
        eps: Small epsilon for numerical stability

    Returns:
        SIGReg loss value (scalar)
    """
    B, D = embeddings.shape

    if B < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Standardize embeddings
    mean = embeddings.mean(dim=0, keepdim=True)
    std = embeddings.std(dim=0, keepdim=True) + eps
    z = (embeddings - mean) / std

    # Create and apply LEJEPA test
    univariate_test = EppsPulley(t_max=t_max, n_points=n_points)
    test = SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=num_slices,
        reduction="mean",
    )

    return test.to(embeddings.device)(z)
