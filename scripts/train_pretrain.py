#!/usr/bin/env python
"""
Heart-JEPA Pre-training Script

Uses PyTorch Lightning + Hydra following LEJEPA patterns.
Trains the encoder with SIGReg + Invariance loss on unlabeled PCG data.

Usage:
    python scripts/train_pretrain.py
    python scripts/train_pretrain.py ++batch_size=64 ++lr=3e-4
    python scripts/train_pretrain.py ++bstat_lambda=0.05 ++n_views=8
"""

import os
import sys
from pathlib import Path

import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from heart_jepa.models import HeartJEPA
from heart_jepa.losses import SIGReg, invariance_loss
from heart_jepa.data.dataset import PhysioNetDataset
from heart_jepa.data.augmentations import MultiViewTransform


class HeartJEPAPretrainModule(pl.LightningModule):
    """
    Lightning module for Heart-JEPA pre-training.

    Combines SIGReg loss (normality regularization) with
    invariance loss (multi-view consistency).
    """

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
            seg_hidden_dim=cfg.seg_hidden_dim,
            seg_num_classes=cfg.seg_num_classes,
            cls_hidden_dim=cfg.cls_hidden_dim,
            cls_num_classes=cfg.cls_num_classes,
        )

        # SIGReg loss using official LEJEPA
        self.sigreg = SIGReg(
            num_slices=cfg.bstat_num_slices,
            t_max=cfg.bstat_t_max,
            n_points=cfg.bstat_n_points,
            weight=cfg.bstat_lambda,
        )

        # Invariance loss weight
        self.invariance_weight = cfg.invariance_weight
        self.invariance_temp = cfg.invariance_temp

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        views, _ = batch  # (B, V, C, H, W), labels ignored for pretraining

        # Get projections
        proj, _, _ = self.model(views)  # proj: (B, V, proj_dim)

        # SIGReg loss - encourages Gaussian distribution
        loss_sigreg = self.sigreg(proj)

        # Invariance loss - encourages view consistency
        loss_inv = invariance_loss(proj, temperature=self.invariance_temp)
        loss_inv = self.invariance_weight * loss_inv

        # Total loss
        loss = loss_sigreg + loss_inv

        # Logging
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/sigreg", loss_sigreg)
        self.log("train/invariance", loss_inv)

        return loss

    def validation_step(self, batch, batch_idx):
        views, _ = batch

        proj, _, _ = self.model(views)

        loss_sigreg = self.sigreg(proj)
        loss_inv = self.invariance_weight * invariance_loss(
            proj, temperature=self.invariance_temp
        )
        loss = loss_sigreg + loss_inv

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/sigreg", loss_sigreg, sync_dist=True)
        self.log("val/invariance", loss_inv, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # Optimizer
        if self.cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.lr,
            )

        # Scheduler
        if self.cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.max_epochs,
                eta_min=self.cfg.lr * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        return optimizer


class HeartJEPADataModule(pl.LightningDataModule):
    """Lightning data module for Heart-JEPA."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.transform = MultiViewTransform(n_views=cfg.n_views)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = PhysioNetDataset(
                self.cfg.data_dir,
                split="train",
                transform=self.transform,
            )
            self.val_dataset = PhysioNetDataset(
                self.cfg.data_dir,
                split="val",
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    print(OmegaConf.to_yaml(cfg))

    # Seed
    pl.seed_everything(cfg.seed)

    # Model
    model = HeartJEPAPretrainModule(cfg)

    # Data
    datamodule = HeartJEPADataModule(cfg)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename="heart-jepa-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=cfg.save_top_k,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    if cfg.wandb_project:
        logger = WandbLogger(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=OmegaConf.to_container(cfg),
        )
    else:
        logger = TensorBoardLogger("tb_logs", name="heart-jepa")

    # Trainer with distributed training support
    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        precision=cfg.precision,
        sync_batchnorm=cfg.sync_batchnorm if cfg.devices != 1 else False,
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        gradient_clip_val=1.0,
        # Distributed training optimizations
        use_distributed_sampler=True,  # Automatically shard data across GPUs
    )

    # Train
    trainer.fit(model, datamodule)

    print(f"Best model saved at: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
