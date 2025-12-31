"""Invariance loss for multi-view learning."""

import torch
import torch.nn.functional as F


def invariance_loss(
    embeddings: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Compute invariance loss between views of the same sample.

    Encourages embeddings of different views of the same sample to be similar.

    Args:
        embeddings: Embedding tensor of shape (B, V, D) where V is number of views
        temperature: Temperature for softmax scaling

    Returns:
        Invariance loss (scalar)
    """
    B, V, D = embeddings.shape

    if V < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1)

    # Compute pairwise cosine similarity within each sample
    # embeddings: (B, V, D) -> similarity: (B, V, V)
    similarity = torch.bmm(embeddings, embeddings.transpose(1, 2))

    # Scale by temperature
    similarity = similarity / temperature

    # Create mask for positive pairs (different views of same sample)
    # All off-diagonal elements are positive pairs
    mask = ~torch.eye(V, dtype=torch.bool, device=embeddings.device)
    mask = mask.unsqueeze(0).expand(B, -1, -1)

    # InfoNCE-style loss: maximize similarity of positive pairs
    # For each view, other views of same sample are positives
    loss = 0.0
    for i in range(V):
        # Similarity of view i to all other views
        sim_i = similarity[:, i, :]  # (B, V)

        # Log-softmax and take positive pairs
        log_prob = F.log_softmax(sim_i, dim=-1)

        # Average over positive pairs (other views)
        pos_mask = mask[:, i, :]  # (B, V)
        loss -= (log_prob * pos_mask.float()).sum(dim=-1) / pos_mask.sum(dim=-1)

    loss = loss.mean() / V

    return loss


class InvarianceLoss(torch.nn.Module):
    """
    Invariance loss module for multi-view learning.

    Encourages representations of different views to be similar.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        weight: float = 1.0,
    ):
        """
        Args:
            temperature: Temperature for softmax scaling
            weight: Loss weight multiplier
        """
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute invariance loss.

        Args:
            embeddings: Embedding tensor of shape (B, V, D)

        Returns:
            Weighted invariance loss
        """
        loss = invariance_loss(embeddings, temperature=self.temperature)
        return self.weight * loss
