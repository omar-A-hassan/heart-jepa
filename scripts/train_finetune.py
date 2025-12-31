#!/usr/bin/env python
"""
Heart-JEPA Fine-tuning Script

Uses PyTorch Lightning + Hydra for fine-tuning on classification or segmentation.
Load pretrained weights and train task-specific heads.

Usage:
    python scripts/train_finetune.py task=classification
    python scripts/train_finetune.py task=segmentation
    python scripts/train_finetune.py task=classification ++freeze_encoder=true
"""

import os
import sys
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from heart_jepa.models import HeartJEPA
from heart_jepa.losses import SegmentationLoss
from heart_jepa.data.dataset import PhysioNetDataset, SegmentationDataset
from heart_jepa.data.augmentations import TestTransform


class HeartJEPAClassificationModule(pl.LightningModule):
    """Lightning module for Heart-JEPA classification fine-tuning."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # Model
        self.model = HeartJEPA(
            backbone=cfg.backbone,
            pretrained=cfg.pretrained,
            proj_dim=cfg.proj_dim,
            hidden_dim=cfg.hidden_dim,
            cls_num_classes=cfg.cls_num_classes,
            cls_hidden_dim=cfg.cls_hidden_dim,
            cls_dropout=cfg.cls_dropout,
        )

        # Load pretrained weights if provided
        if cfg.pretrained_checkpoint:
            self._load_pretrained(cfg.pretrained_checkpoint)

        # Freeze encoder if specified
        if cfg.freeze_encoder:
            self.model.freeze_encoder()
            print("Encoder frozen - only training classification head")

        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=cfg.cls_num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=cfg.cls_num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=cfg.cls_num_classes, average="macro")

    def _load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Load with strict=False to allow missing/extra keys
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")
        if missing:
            print(f"  Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}...")

    def forward(self, x):
        return self.model.forward_classification(x)

    def training_step(self, batch, batch_idx):
        specs, labels = batch
        logits = self(specs)
        loss = self.criterion(logits, labels)

        self.train_acc(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        specs, labels = batch
        logits = self(specs)
        loss = self.criterion(logits, labels)

        self.val_acc(logits, labels)
        self.val_f1(logits, labels)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, prog_bar=True, sync_dist=True)
        self.log("val/f1", self.val_f1, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Only optimize unfrozen parameters
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.finetune_lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.finetune_epochs,
            eta_min=self.cfg.finetune_lr * 0.01,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class HeartJEPASegmentationModule(pl.LightningModule):
    """Lightning module for Heart-JEPA segmentation fine-tuning."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # Model
        self.model = HeartJEPA(
            backbone=cfg.backbone,
            pretrained=cfg.pretrained,
            proj_dim=cfg.proj_dim,
            hidden_dim=cfg.hidden_dim,
            seg_num_classes=cfg.seg_num_classes,
            seg_hidden_dim=cfg.seg_hidden_dim,
            seg_output_frames=cfg.seg_output_frames,
        )

        # Load pretrained weights if provided
        if cfg.pretrained_checkpoint:
            self._load_pretrained(cfg.pretrained_checkpoint)

        # Freeze encoder if specified
        if cfg.freeze_encoder:
            self.model.freeze_encoder()
            print("Encoder frozen - only training segmentation head")

        # Loss
        self.criterion = SegmentationLoss(
            ce_weight=cfg.seg_ce_weight,
            dice_weight=cfg.seg_dice_weight,
            num_classes=cfg.seg_num_classes,
        )

    def _load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    def forward(self, x):
        return self.model.forward_segmentation(x)

    def training_step(self, batch, batch_idx):
        specs, labels = batch  # labels: (B, T) frame-level
        logits = self(specs)  # (B, C, T)
        loss = self.criterion(logits, labels)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        specs, labels = batch
        logits = self(specs)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = logits.argmax(dim=1)  # (B, T)
        acc = (preds == labels).float().mean()

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.finetune_lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.finetune_epochs,
            eta_min=self.cfg.finetune_lr * 0.01,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class FinetuneDataModule(pl.LightningDataModule):
    """Lightning data module for fine-tuning."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.transform = TestTransform()  # Single view for fine-tuning
        self.task = cfg.get("task", "classification")

    def setup(self, stage=None):
        if self.task == "segmentation":
            # Use SegmentationDataset with pseudo-labels
            DatasetClass = SegmentationDataset
            extra_kwargs = {"output_frames": self.cfg.seg_output_frames}
        else:
            # Use PhysioNetDataset for classification
            DatasetClass = PhysioNetDataset
            extra_kwargs = {}

        if stage == "fit" or stage is None:
            self.train_dataset = DatasetClass(
                self.cfg.data_dir,
                split="train",
                transform=self.transform,
                **extra_kwargs,
            )
            self.val_dataset = DatasetClass(
                self.cfg.data_dir,
                split="val",
                transform=self.transform,
                **extra_kwargs,
            )
        if stage == "test" or stage is None:
            self.test_dataset = DatasetClass(
                self.cfg.data_dir,
                split="test",
                transform=self.transform,
                **extra_kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.finetune_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.finetune_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.finetune_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Main fine-tuning function."""
    print(OmegaConf.to_yaml(cfg))

    # Seed
    pl.seed_everything(cfg.seed)

    # Select task
    task = cfg.get("task", "classification")
    print(f"\nFine-tuning task: {task}")

    # Model
    if task == "classification":
        model = HeartJEPAClassificationModule(cfg)
        monitor_metric = "val/acc"
        mode = "max"
    elif task == "segmentation":
        model = HeartJEPASegmentationModule(cfg)
        monitor_metric = "val/loss"
        mode = "min"
    else:
        raise ValueError(f"Unknown task: {task}")

    # Data
    datamodule = FinetuneDataModule(cfg)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename=f"heart-jepa-{task}-{{epoch:02d}}-{{val/acc:.4f}}",
            monitor=monitor_metric,
            mode=mode,
            save_top_k=cfg.save_top_k,
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=monitor_metric,
            patience=10,
            mode=mode,
        ),
    ]

    # Logger
    if cfg.wandb_project:
        logger = WandbLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"finetune-{task}",
            config=OmegaConf.to_container(cfg),
        )
    else:
        logger = TensorBoardLogger("tb_logs", name=f"heart-jepa-{task}")

    # Trainer
    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        max_epochs=cfg.finetune_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=1.0,
    )

    # Train
    trainer.fit(model, datamodule)

    # Test
    trainer.test(model, datamodule)

    print(f"\nBest model saved at: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
