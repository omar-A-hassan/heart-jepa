"""Attention map extraction from ViT encoder."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union
from contextlib import contextmanager


class AttentionExtractor:
    """
    Extract attention maps from Vision Transformer models.

    Uses forward hooks to capture attention weights from transformer blocks.
    Works with timm ViT models.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Initialize attention extractor.

        Args:
            model: ViT model (or HeartJEPA model containing ViT)
            layer_indices: Which layers to extract attention from.
                          None = all layers, [-1] = last layer only
        """
        self.model = model
        self.layer_indices = layer_indices
        self.attention_maps: List[torch.Tensor] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Find the ViT backbone
        self.vit = self._find_vit_backbone(model)
        if self.vit is None:
            raise ValueError("Could not find ViT backbone in model")

        # Get attention modules
        self.attention_modules = self._get_attention_modules()

    def _find_vit_backbone(self, model: nn.Module) -> Optional[nn.Module]:
        """Find the timm ViT backbone in the model hierarchy."""
        # Direct ViT model
        if hasattr(model, 'blocks'):
            return model

        # HeartJEPA -> encoder -> encoder -> backbone
        if hasattr(model, 'encoder'):
            encoder = model.encoder
            if hasattr(encoder, 'encoder'):
                inner = encoder.encoder
                if hasattr(inner, 'backbone'):
                    return inner.backbone
                return inner
            if hasattr(encoder, 'backbone'):
                return encoder.backbone
            return encoder

        # HeartJEPAEncoder -> encoder -> backbone
        if hasattr(model, 'backbone'):
            return model.backbone

        return None

    def _get_attention_modules(self) -> List[nn.Module]:
        """Get list of attention modules from ViT blocks."""
        modules = []

        if hasattr(self.vit, 'blocks'):
            for block in self.vit.blocks:
                if hasattr(block, 'attn'):
                    modules.append(block.attn)

        return modules

    def _attention_hook(self, module: nn.Module, input: Tuple, output: torch.Tensor):
        """Hook function to capture attention weights."""
        # timm's Attention module stores attention weights in forward
        # We need to recompute them from Q, K
        # The hook captures the output, but we need attention weights
        pass

    def _create_attention_hook(self, layer_idx: int):
        """Create a hook that captures attention for a specific layer."""
        def hook(module, input, output):
            # For timm ViT, we need to capture attention during forward
            # The attention is computed inside the module
            # We'll store a placeholder and compute attention separately
            self.attention_maps.append((layer_idx, output))
        return hook

    def register_hooks(self):
        """Register forward hooks on attention modules."""
        self.clear_hooks()
        self.attention_maps = []

        indices = self.layer_indices
        if indices is None:
            indices = list(range(len(self.attention_modules)))
        else:
            # Handle negative indices
            indices = [i if i >= 0 else len(self.attention_modules) + i for i in indices]

        for idx in indices:
            if 0 <= idx < len(self.attention_modules):
                module = self.attention_modules[idx]
                hook = module.register_forward_hook(self._create_attention_hook(idx))
                self.hooks.append(hook)

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = []

    @contextmanager
    def capture(self):
        """Context manager for capturing attention maps."""
        self.register_hooks()
        try:
            yield
        finally:
            self.clear_hooks()

    def get_attention(
        self,
        x: torch.Tensor,
        aggregate: str = "mean",
    ) -> torch.Tensor:
        """
        Get attention maps for input.

        This method computes attention by doing a forward pass with
        attention weight extraction enabled.

        Args:
            x: Input tensor of shape (B, C, H, W)
            aggregate: How to aggregate attention across heads
                      "mean" = average across heads
                      "max" = max across heads
                      None = return all heads

        Returns:
            Attention maps of shape:
            - (B, N, N) if aggregate is "mean" or "max"
            - (B, num_heads, N, N) if aggregate is None
            Where N = num_patches + 1 (including CLS token)
        """
        # Use the dedicated attention extraction method
        return get_attention_maps(
            self.model,
            x,
            layer_idx=self.layer_indices[-1] if self.layer_indices else -1,
            aggregate=aggregate,
        )


def get_attention_maps(
    model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = -1,
    aggregate: str = "mean",
) -> torch.Tensor:
    """
    Extract attention maps from a ViT model.

    This function uses a more direct approach by modifying the forward
    pass to return attention weights.

    Args:
        model: ViT model or HeartJEPA model
        x: Input tensor of shape (B, C, H, W)
        layer_idx: Which layer to extract from (-1 = last)
        aggregate: "mean", "max", or None

    Returns:
        Attention maps tensor
    """
    # Find the ViT backbone
    vit = _find_vit(model)
    if vit is None:
        raise ValueError("Could not find ViT backbone")

    # Store original forward methods
    original_forwards = {}
    attention_storage = {}

    def make_attn_forward(block_idx, original_attn):
        """Create a new forward that stores attention weights."""
        def new_forward(x, attn_mask=None):
            B, N, C = x.shape

            # Get qkv
            qkv = original_attn.qkv(x)
            qkv = qkv.reshape(B, N, 3, original_attn.num_heads, C // original_attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
            q, k, v = qkv.unbind(0)

            # Compute attention weights
            scale = (C // original_attn.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale

            # Apply attention mask if provided
            if attn_mask is not None:
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)

            # Store attention weights
            attention_storage[block_idx] = attn.detach()

            # Apply attention dropout and compute output
            attn = original_attn.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = original_attn.proj(x)
            x = original_attn.proj_drop(x)

            return x
        return new_forward

    # Patch attention modules
    blocks = vit.blocks if hasattr(vit, 'blocks') else []
    for idx, block in enumerate(blocks):
        if hasattr(block, 'attn'):
            original_forwards[idx] = block.attn.forward
            block.attn.forward = make_attn_forward(idx, block.attn)

    try:
        # Forward pass
        with torch.no_grad():
            if hasattr(model, 'encoder'):
                # HeartJEPA model
                if hasattr(model.encoder, 'encoder'):
                    model.encoder.encoder.forward_features(x)
                else:
                    model.encoder.forward_features(x)
            elif hasattr(model, 'forward_features'):
                model.forward_features(x)
            else:
                model(x)

        # Get requested layer's attention
        num_layers = len(blocks)
        target_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx

        if target_idx not in attention_storage:
            raise ValueError(f"Layer {layer_idx} not found in attention storage")

        attn = attention_storage[target_idx]  # (B, heads, N, N)

        # Aggregate across heads
        if aggregate == "mean":
            attn = attn.mean(dim=1)  # (B, N, N)
        elif aggregate == "max":
            attn = attn.max(dim=1).values  # (B, N, N)
        # else: return all heads (B, heads, N, N)

        return attn

    finally:
        # Restore original forwards
        for idx, original in original_forwards.items():
            blocks[idx].attn.forward = original


def _find_vit(model: nn.Module) -> Optional[nn.Module]:
    """Find the timm ViT backbone in the model hierarchy."""
    if hasattr(model, 'blocks'):
        return model

    if hasattr(model, 'encoder'):
        encoder = model.encoder
        if hasattr(encoder, 'encoder'):
            inner = encoder.encoder
            if hasattr(inner, 'backbone'):
                return inner.backbone
            if hasattr(inner, 'blocks'):
                return inner
        if hasattr(encoder, 'backbone'):
            return encoder.backbone
        if hasattr(encoder, 'blocks'):
            return encoder

    if hasattr(model, 'backbone'):
        backbone = model.backbone
        if hasattr(backbone, 'blocks'):
            return backbone

    return None


def attention_to_temporal(
    attention: torch.Tensor,
    output_frames: int,
    use_cls: bool = True,
) -> torch.Tensor:
    """
    Convert 2D patch attention to 1D temporal saliency.

    Args:
        attention: Attention maps of shape (B, N, N) or (B, heads, N, N)
            where N = num_patches + 1 (CLS + patches)
        output_frames: Number of output temporal frames
        use_cls: If True, use attention FROM CLS token to patches
                 If False, use mean attention between all patches

    Returns:
        Temporal saliency of shape (B, output_frames)
    """
    # Handle multi-head case
    if attention.dim() == 4:
        attention = attention.mean(dim=1)  # (B, N, N)

    B, N, _ = attention.shape
    num_patches = N - 1  # Exclude CLS token

    if use_cls:
        # Attention from CLS token to all patches
        # CLS is at position 0
        saliency = attention[:, 0, 1:]  # (B, num_patches)
    else:
        # Mean attention received by each patch (excluding CLS)
        saliency = attention[:, 1:, 1:].mean(dim=1)  # (B, num_patches)

    # Reshape patches to 2D grid (assuming square)
    # For ViT-B/16 with 224x224 input: 14x14 = 196 patches
    grid_size = int(num_patches ** 0.5)
    if grid_size * grid_size != num_patches:
        # Non-square, just use 1D
        grid_size = num_patches
        saliency_2d = saliency.unsqueeze(1)  # (B, 1, num_patches)
    else:
        saliency_2d = saliency.view(B, grid_size, grid_size)  # (B, H, W)
        # For spectrograms, temporal dimension is typically width
        # Average across frequency (height) to get temporal saliency
        saliency_1d = saliency_2d.mean(dim=1)  # (B, W)
        saliency = saliency_1d

    # Interpolate to output_frames
    saliency = saliency.unsqueeze(1)  # (B, 1, T)
    saliency = torch.nn.functional.interpolate(
        saliency,
        size=output_frames,
        mode='linear',
        align_corners=False,
    )
    saliency = saliency.squeeze(1)  # (B, output_frames)

    return saliency
