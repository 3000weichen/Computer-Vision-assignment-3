"""
Improved baseline for Jester:
- Stronger data augmentation
- Dropout in classification head
- AdamW with weight decay
- Early stopping on validation loss
- Learning rate scheduler
"""

import os
import random
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt


# ----------------------
# Config
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class CONFIG:
    TRAIN_CSV   = os.path.join(BASE_DIR, "jester-v1-small-train.csv")   # or train.csv
    VAL_CSV     = os.path.join(BASE_DIR, "jester-v1-validation.csv")
    LABELS_CSV  = os.path.join(BASE_DIR, "jester-v1-labels.csv")
    FRAME_ROOT  = os.path.join(BASE_DIR, "20bn-jester-v1")

    NUM_CLASSES = 27
    MAX_TRAIN_SAMPLES = 10000     # data limiting

    BATCH_SIZE  = 32
    NUM_WORKERS = 4
    NUM_EPOCHS  = 30             # use more epochs for early stopping
    LR          = 5e-4
    WEIGHT_DECAY = 1e-3          # normalization for AdamW
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

    IMG_SIZE = 112
    SEED     = 42

    EARLY_STOP_PATIENCE = 5      # early stopping patience


# ----------------------
# Reproducibility
# ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG.SEED)


# ----------------------
# Label mapping
# ----------------------
def load_label_mapping(labels_csv: str):
    labels = pd.read_csv(labels_csv, header=None)[0].tolist()
    label2idx = {name: i for i, name in enumerate(labels)}
    return label2idx, labels


# ----------------------
# Dataset
# ----------------------
class JesterFrameFolderDataset(Dataset):
    """
    CSV 每行: "video_id;label_name"
    对应帧目录: FRAME_ROOT/<video_id>/*.jpg
    """

    def __init__(
        self,
        csv_path: str,
        frame_root: str,
        label2idx: dict,
        transform: Optional[object] = None,
        max_samples: Optional[int] = None,
        is_train: bool = True,
    ):
        self.frame_root = frame_root
        self.label2idx = label2idx
        self.transform = transform
        self.is_train = is_train

        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] == 1:
            df = df[0].str.split(";", expand=True)
        df.columns = ["video_id", "label_name"]

        df["video_id"] = df["video_id"].astype(str).str.strip()
        df["label_name"] = df["label_name"].astype(str).str.strip()

        available_ids = {
            name
            for name in os.listdir(frame_root)
            if os.path.isdir(os.path.join(frame_root, name)) and name.isdigit()
        }

        before = len(df)
        df = df[df["video_id"].isin(available_ids)].reset_index(drop=True)
        missing = before - len(df)
        if missing > 0:
            print(
                f"[WARN] {missing} samples from {os.path.basename(csv_path)} "
                f"do not exist under {os.path.basename(frame_root)} and will be skipped."
            )

        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=CONFIG.SEED).reset_index(drop=True)

        self.samples = df

    def __len__(self):
        return len(self.samples)

    def _get_video_dir(self, video_id: str) -> str:
        return os.path.join(self.frame_root, str(video_id))

    def _load_middle_frame_from_folder(self, video_dir: str):
        frame_files = sorted(
            [f for f in os.listdir(video_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        if not frame_files:
            raise RuntimeError(f"No frames found in {video_dir}")

        mid_idx = len(frame_files) // 2
        frame_path = os.path.join(video_dir, frame_files[mid_idx])

        img = cv2.imread(frame_path)
        if img is None:
            raise RuntimeError(f"Cannot read frame {frame_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        video_id = row["video_id"]
        label_name = row["label_name"]

        video_dir = self._get_video_dir(video_id)
        img = self._load_middle_frame_from_folder(video_dir)

        if self.transform is not None:
            img = self.transform(img)

        label_idx = self.label2idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)
        return img, label


# ----------------------
# Transforms for data augmentation (stronger for training)
# ----------------------
def get_transforms(is_train: bool = True):
    if is_train:
        ops = [
            T.ToPILImage(),
            T.RandomResizedCrop(CONFIG.IMG_SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            # Color jitter for stronger augmentation, to avoid overfitting
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]
    else:
        ops = [
            T.ToPILImage(),
            T.Resize((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]
    return T.Compose(ops)


# ----------------------
# Model (add Dropout in head)
# ----------------------
class GestureImprovedNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features

        # remove final fc layer
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),               # use dropout to avoid overfitting
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


# ----------------------
# Train / Eval
# ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ----------------------
# Plot helpers
# ----------------------
def plot_history(history, prefix: str, out_dir: str = BASE_DIR):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix} loss curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix.lower().replace(' ', '_')}_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["val_acc"], label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{prefix} accuracy curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix.lower().replace(' ', '_')}_acc.png"))
    plt.close()


# ----------------------
# Main with early stopping + LR scheduler
# ----------------------
def main():
    label2idx, idx2label = load_label_mapping(CONFIG.LABELS_CSV)

    train_dataset = JesterFrameFolderDataset(
        csv_path=CONFIG.TRAIN_CSV,
        frame_root=CONFIG.FRAME_ROOT,
        label2idx=label2idx,
        transform=get_transforms(is_train=True),
        max_samples=CONFIG.MAX_TRAIN_SAMPLES,
        is_train=True,
    )
    val_dataset = JesterFrameFolderDataset(
        csv_path=CONFIG.VAL_CSV,
        frame_root=CONFIG.FRAME_ROOT,
        label2idx=label2idx,
        transform=get_transforms(is_train=False),
        max_samples=None,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=True,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=False,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)} (data-limited)")
    print(f"Val samples:   {len(val_dataset)}")

    device = CONFIG.DEVICE
    model = GestureImprovedNet(CONFIG.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.LR, weight_decay=CONFIG.WEIGHT_DECAY
    )
   
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=True,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, CONFIG.NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{CONFIG.NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        # Early stopping  val loss
        if val_loss < best_val_loss - 1e-3:   # small epsilon to avoid float precision issue
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "improved_best.pth"))
            print(f"  -> New best val loss {best_val_loss:.4f}, acc {best_val_acc:.4f}, model saved.")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= CONFIG.EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    plot_history(history, prefix="Improved baseline", out_dir=BASE_DIR)
    print("Saved curves to 'improved_baseline_loss.png' and 'improved_baseline_acc.png'.")


if __name__ == "__main__":
    main()