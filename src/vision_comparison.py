"""Six architectures adapted from the original coursework notebook.
No model weights or datasets are loaded during import. See ATTRIBUTION.md.
"""
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MODEL_NAMES = ("CNN2D", "CNN3D (RGB depth)", "RNN (ResNet features)",
               "LSTM (ResNet features)", "ResNet18 feature extractor", "ResNet18 partial fine-tune")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))

class CNN2DImproved(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            ResidualBlock2D(32, 64, stride=2),
            ResidualBlock2D(64, 128, stride=2),
            ResidualBlock2D(128, 192, stride=2),
            ResidualBlock2D(192, 192),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(192, 2),
        )

    def forward(self, x):
        return self.features(x)

class CNN3DImproved(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 12, 3, padding=1, bias=False), nn.BatchNorm3d(12), nn.SiLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(12, 24, 3, padding=1, bias=False), nn.BatchNorm3d(24), nn.SiLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(24, 48, 3, padding=1, bias=False), nn.BatchNorm3d(48), nn.SiLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(48, 64, 3, padding=1, bias=False), nn.BatchNorm3d(64), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.features(x.unsqueeze(1))

class PretrainedRowSequence(nn.Module):
    """Frozen ResNet spatial features become a short sequence of seven rows."""

    def __init__(self, kind: str, weights="imagenet"):
        super().__init__()
        if kind not in {"RNN", "LSTM"}:
            raise ValueError("kind must be RNN or LSTM")
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights == "imagenet" else None)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        recurrent = nn.RNN if kind == "RNN" else nn.LSTM
        self.recurrent = recurrent(512, 128, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.25), nn.Linear(256, 2))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        rows = features.mean(dim=3).transpose(1, 2)
        outputs, _ = self.recurrent(rows)
        return self.head(outputs.mean(dim=1))


def seed_all(seed, deterministic=True):
    """Seed a new run; deterministic operations are requested by default."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def make_transforms(size):
    if size not in (128, 224):
        raise ValueError("The original protocols use image sizes 128 or 224")
    mean, std = (MEAN, STD) if size == 224 else ([0.5] * 3, [0.5] * 3)
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    validation = transforms.Compose([
        transforms.Resize(round(size * 256 / 224)), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    return train, validation


def make_loaders(data_dir, size, batch_size=16, device="cpu"):
    train_tf, val_tf = make_transforms(size)
    root = Path(data_dir)
    parts = {phase: datasets.ImageFolder(root / phase, transform=tf)
             for phase, tf in (("train", train_tf), ("val", val_tf))}
    if parts["train"].classes != ["ants", "bees"] or parts["val"].classes != ["ants", "bees"]:
        raise ValueError("Expected matching ants/bees classes in train and val")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return {phase: DataLoader(ds, batch_size=batch_size, shuffle=phase == "train",
                              num_workers=0, pin_memory=torch.device(device).type == "cuda")
            for phase, ds in parts.items()}


def build_model(name, weights="imagenet"):
    """Set weights=None for offline architecture checks; no automatic downloads then."""
    if weights not in (None, "imagenet"):
        raise ValueError("weights must be None or imagenet")
    if name == "CNN2D":
        return CNN2DImproved(), None
    if name == "CNN3D (RGB depth)":
        return CNN3DImproved(), None
    if name in ("RNN (ResNet features)", "LSTM (ResNet features)"):
        return PretrainedRowSequence(name.split()[0], weights=weights), "sequence"
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown architecture: {name}")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights else None)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if name == "ResNet18 feature extractor":
        model.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(512, 128), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(128, 2))
        return model, "feature"
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))
    return model, "partial"


def set_training_mode(model, training, freeze_mode=None):
    """Frozen batch-normalization statistics must stay fixed during head training."""
    model.train(training)
    if training and freeze_mode == "feature":
        model.eval()
        model.fc.train()
    elif training and freeze_mode == "partial":
        model.eval()
        model.layer4.train()
        model.fc.train()
    elif training and freeze_mode == "sequence":
        model.backbone.eval()


def make_optimizer(model, freeze_mode):
    if freeze_mode == "partial":
        return torch.optim.AdamW([
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 7e-4}], weight_decay=1e-4)
    rate = 8e-4 if freeze_mode == "feature" else 1e-3
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=rate, weight_decay=1e-4)


def fit(model, loaders, optimizer, epochs, device="cpu", freeze_mode=None):
    """Select a checkpoint on validation; returned metrics are validation metrics."""
    if epochs < 1 or any(len(loaders[phase].dataset) == 0 for phase in ("train", "val")):
        raise ValueError("Positive epochs and nonempty train/validation datasets are required")
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best = {"accuracy": -1.0, "loss": float("inf"), "epoch": 0, "weights": None}
    history = []
    for epoch in range(1, epochs + 1):
        row = {"epoch": epoch}
        for phase in ("train", "val"):
            training = phase == "train"
            set_training_mode(model, training, freeze_mode)
            loss_sum = correct = seen = 0
            for images, labels in loaders[phase]:
                images, labels = images.to(device), labels.to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    logits = model(images)
                    loss = criterion(logits, labels)
                    if training:
                        loss.backward()
                        optimizer.step()
                seen += labels.size(0)
                loss_sum += loss.item() * labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
            row[f"{phase}_loss"] = loss_sum / seen
            row[f"{phase}_accuracy"] = correct / seen
        scheduler.step()
        history.append(row)
        if (row["val_accuracy"], -row["val_loss"]) > (best["accuracy"], -best["loss"]):
            best = dict(accuracy=row["val_accuracy"], loss=row["val_loss"], epoch=epoch,
                        weights=deepcopy(model.state_dict()))
    model.load_state_dict(best["weights"])
    model.eval()
    return model, history, best["epoch"]
