"""
notebooks/train_vision_model.py
--------------------------------
Training script for the DenseNet121 medical image classifier.

Supports: NIH Chest X-ray · CheXpert · MIMIC-CXR (after access is granted)

Usage:
  python notebooks/train_vision_model.py \
      --data_dir data/medical_images/nih_chest_xray \
      --labels_csv data/medical_images/nih_chest_xray/Data_Entry_2017.csv \
      --output_dir models/vision_model \
      --epochs 30 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    import torchvision.transforms as T
    import torchvision.models as models
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from loguru import logger


# ── NIH CheXpert 14-class labels ──────────────────────────────────────────────

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "No Finding",
]


# ── Dataset ───────────────────────────────────────────────────────────────────

class NIHChestXrayDataset(Dataset):
    """
    NIH Chest X-ray Dataset loader.

    Expects:
      - images in data_dir/images/
      - labels CSV with columns: Image Index, Finding Labels
    """

    def __init__(
        self,
        data_dir: str,
        labels_csv: str,
        labels: list[str],
        transform=None,
        max_samples: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.labels = labels
        self.transform = transform

        df = pd.read_csv(labels_csv)
        if max_samples:
            df = df.head(max_samples)

        self.image_paths = df["Image Index"].tolist()
        self.targets = self._encode_labels(df["Finding Labels"].tolist())

        logger.info(f"Dataset loaded: {len(self.image_paths)} samples")

    def _encode_labels(self, finding_labels: list[str]) -> np.ndarray:
        """Convert pipe-separated label strings to multi-hot encoding."""
        n = len(finding_labels)
        k = len(self.labels)
        matrix = np.zeros((n, k), dtype=np.float32)
        for i, label_str in enumerate(finding_labels):
            for label in label_str.split("|"):
                label = label.strip()
                if label in self.labels:
                    matrix[i, self.labels.index(label)] = 1.0
        return matrix

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.data_dir / "images" / self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Return blank image if file not found (graceful degradation)
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, target


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(image_size: int = 224, augment: bool = True):
    """Build training and validation transforms."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        train_transform = T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        train_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_transform, val_transform


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(num_classes: int, pretrained: bool = True) -> "nn.Module":
    model = models.densenet121(weights="DEFAULT" if pretrained else None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
        nn.Sigmoid(),
    )
    return model


# ── Training loop ─────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model,
        device,
        output_dir: Path,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, factor=0.5, verbose=True
        )
        self.best_auc = 0.0
        self.history: list[dict] = []

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        for batch_idx, (images, targets) in enumerate(loader):
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                logger.info(f"  batch {batch_idx}/{len(loader)} | loss={loss.item():.4f}")
        return {"train_loss": total_loss / len(loader)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_probs, all_targets = [], []
        total_loss = 0.0
        for images, targets in loader:
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            all_probs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        probs = np.vstack(all_probs)
        targets = np.vstack(all_targets)

        # Per-class AUC
        aucs = []
        for i in range(targets.shape[1]):
            if targets[:, i].sum() > 0:
                try:
                    auc = roc_auc_score(targets[:, i], probs[:, i])
                    aucs.append(auc)
                except Exception:
                    pass
        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        return {
            "val_loss": total_loss / len(loader),
            "mean_auc": mean_auc,
            "per_class_auc": aucs,
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, path)
        if is_best:
            best_path = self.output_dir / "densenet121_best.pth"
            torch.save(self.model.state_dict(), best_path)
            logger.success(f"New best model saved: AUC={metrics['mean_auc']:.4f}")

    def train(self, train_loader, val_loader, epochs: int):
        logger.info(f"Starting training | epochs={epochs} | device={self.device}")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            elapsed = time.time() - t0

            metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.history.append(metrics)
            self.scheduler.step(val_metrics["val_loss"])

            is_best = val_metrics["mean_auc"] > self.best_auc
            if is_best:
                self.best_auc = val_metrics["mean_auc"]

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"mean_AUC={val_metrics['mean_auc']:.4f} | "
                f"time={elapsed:.0f}s"
            )
            self.save_checkpoint(epoch, metrics, is_best)

        # Save training history
        history_path = self.output_dir / "training_history.json"
        history_path.write_text(json.dumps(self.history, indent=2))
        logger.success(f"Training complete. Best AUC: {self.best_auc:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DenseNet121 on NIH Chest X-rays")
    parser.add_argument("--data_dir", required=True, help="Path to image directory")
    parser.add_argument("--labels_csv", required=True, help="Path to labels CSV")
    parser.add_argument("--output_dir", default="models/vision_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap dataset size")
    parser.add_argument("--no_pretrain", action="store_true", help="Train from scratch")
    args = parser.parse_args()

    if not _TORCH_AVAILABLE:
        logger.error("PyTorch not installed. pip install torch torchvision")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_transform, val_transform = get_transforms(args.image_size, augment=True)

    full_dataset = NIHChestXrayDataset(
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        labels=NIH_LABELS,
        transform=train_transform,
        max_samples=args.max_samples,
    )

    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = build_model(num_classes=len(NIH_LABELS), pretrained=not args.no_pretrain)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        device=device,
        output_dir=Path(args.output_dir),
        learning_rate=args.lr,
    )
    trainer.train(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
