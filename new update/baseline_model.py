"""
Baseline for Jester: data-limited (5000 videos), single-frame ResNet18 classifier.

Requirements:
- pip install torch torchvision opencv-python pandas
- Adjust the paths in CONFIG below to your local files.
"""

"""
Baseline for Jester: data-limited (5000 videos), single-frame ResNet18 classifier.
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
# import torchvision.models as models   # 不再需要
from torchvision.models import resnet18, ResNet18_Weights  # <<< 新写法
import matplotlib.pyplot as plt        # <<< 用于画图


# ----------------------
# Config（全部相对 baseline_model.py）
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class CONFIG:
    TRAIN_CSV   = os.path.join(BASE_DIR, "jester-v1-small-train.csv")   # 或 train.csv
    VAL_CSV     = os.path.join(BASE_DIR, "jester-v1-validation.csv")
    LABELS_CSV  = os.path.join(BASE_DIR, "jester-v1-labels.csv")
    FRAME_ROOT  = os.path.join(BASE_DIR, "20bn-jester-v1")

    NUM_CLASSES = 27
    MAX_TRAIN_SAMPLES = 10000  # 数据限制

    BATCH_SIZE  = 32
    NUM_WORKERS = 4
    NUM_EPOCHS  = 20
    LR          = 1e-3
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

    IMG_SIZE = 112
    SEED     = 42

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
# Dataset：一条样本 = 一个子文件夹的中间帧
# ----------------------
class JesterFrameFolderDataset(Dataset):
    """
    CSV 每行: "video_id;label_name"
    对应帧目录: FRAME_ROOT/<video_id>/*.jpg
    例如 video_id="4" -> 20bn-jester-v1/4/00001.jpg ...
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

        # 去掉可能的空格
        df["video_id"] = df["video_id"].astype(str).str.strip()
        df["label_name"] = df["label_name"].astype(str).str.strip()

        # 先拿到本地实际有哪些视频文件夹
        available_ids = {
            name
            for name in os.listdir(frame_root)
            if os.path.isdir(os.path.join(frame_root, name)) and name.isdigit()
        }

        # 过滤掉本地不存在的视频
        before = len(df)
        df = df[df["video_id"].isin(available_ids)].reset_index(drop=True)
        missing = before - len(df)
        if missing > 0:
            print(
                f"[WARN] {missing} samples from {os.path.basename(csv_path)} "
                f"do not exist under {os.path.basename(frame_root)} and will be skipped."
            )

        # 再应用 data-limited 抽样
        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=CONFIG.SEED).reset_index(drop=True)

        self.samples = df

    def __len__(self):
        return len(self.samples)

    def _get_video_dir(self, video_id: str) -> str:
        video_dir_name = str(video_id)
        return os.path.join(self.frame_root, video_dir_name)

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
# Transforms
# ----------------------
def get_transforms(is_train: bool = True):
    ops = [
        T.ToPILImage(),
        T.Resize((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE)),
    ]
    if is_train:
        ops += [T.RandomHorizontalFlip()]
    ops += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(ops)


# ----------------------
# ResNet18 baseline
# ----------------------
class GestureBaselineNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # <<< 使用 weights API，去掉 pretrained 警告
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


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
def plot_history(history, out_dir: str = BASE_DIR):
    """绘制 loss 和 accuracy 曲线，并保存为 PNG 文件。"""
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss 曲线
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline loss curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "baseline_loss_curves.png"))
    plt.close()

    # Accuracy 曲线
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["val_acc"], label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline accuracy curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "baseline_accuracy_curves.png"))
    plt.close()


# ----------------------
# Main
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
    model = GestureBaselineNet(CONFIG.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)

    best_val_acc = 0.0

    # <<< 用于画图的历史记录
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, CONFIG.NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{CONFIG.NUM_EPOCHS} "
            f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
            f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "baseline_best.pth"))
            print(f"  -> New best val acc: {best_val_acc:.4f}, model saved.")

    # <<< 训练结束后画图
    plot_history(history, out_dir=BASE_DIR)
    print("Saved curves to 'baseline_loss_curves.png' and 'baseline_accuracy_curves.png'.")


if __name__ == "__main__":
    main()
