"""
训练循环：Trainer 封装训练/验证/checkpoint。
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """创建 MNIST train/test DataLoader。"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


class Trainer:
    """MLP 训练器，封装训练循环和 checkpoint。"""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        self.device = device or _get_device()
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=0  # 由 fit() 设置
        )
        self.best_val_acc = 0.0
        self.metrics_history: list[dict] = []

    def _train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    @torch.inference_mode()
    def _validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        checkpoint_dir: Optional[str] = None,
    ) -> list[dict]:
        """运行完整训练流程，返回 per-epoch 指标列表。"""
        self.scheduler.T_max = epochs
        self.metrics_history = []

        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)

            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": round(train_metrics["loss"], 6),
                "train_acc": round(train_metrics["accuracy"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_acc": round(val_metrics["accuracy"], 6),
            }
            self.metrics_history.append(epoch_metrics)

            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.2%} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.2%}"
            )

            self.scheduler.step()

            if val_metrics["accuracy"] > self.best_val_acc and checkpoint_dir:
                self.best_val_acc = val_metrics["accuracy"]
                self.save_checkpoint(Path(checkpoint_dir) / "best_model.pt")

        return self.metrics_history

    def save_checkpoint(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
            "best_val_acc": self.best_val_acc,
        }, path)

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.metrics_history = checkpoint["metrics_history"]
        self.best_val_acc = checkpoint["best_val_acc"]
