"""Loss functions for Heart-JEPA training."""

from .sigreg import SIGReg, sigreg_loss
from .invariance import invariance_loss
from .segmentation import SegmentationLoss

__all__ = [
    "SIGReg",
    "sigreg_loss",
    "invariance_loss",
    "SegmentationLoss",
]
