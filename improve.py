"""
Improve time series model for Jester:
- Per video use multiple frames (time series) instead of a single frame
- ResNet18 backbone to extract per-frame features (pretrained on ImageNet)
- GRU over time to model temporal dynamics
- AdamW + weight decay + label smoothing + ReduceLROnPlateau + early stopping

Usage (in the assignment folder, with .venv311 activated):
    python improve_times_series.py
"""

import os
import random
from typing import Optional, List

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
    # CSV + frames
    TRAIN_CSV   = os.path.join(BASE_DIR, "jester-v1-small-train.csv")
    VAL_CSV     = os.path.join(BASE_DIR, "jester-v1-validation.csv")
    LABELS_CSV  = os.path.join(BASE_DIR, "jester-v1-labels.csv")
    FRAME_ROOT  = os.path.join(BASE_DIR, "20bn-jester-v1")

    NUM_CLASSES = 27
    MAX_TRAIN_SAMPLES = 10000     # 数据限制
    FRAMES_PER_CLIP = 6          # 每个视频采样 8 帧

    BATCH_SIZE  = 64            # 时序模型显存更吃紧，batch 稍微小一点
    NUM_WORKERS = 16
    NUM_EPOCHS  = 25
    LR          = 3e-4
    WEIGHT_DECAY = 1e-4
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

    IMG_SIZE = 112
    SEED     = 42

    EARLY_STOP_PATIENCE = 4      # 连续 4 个 epoch val loss 无提升则停止
    LABEL_SMOOTHING = 0.1


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
# Dataset：一条样本 = 一个子文件夹的多帧序列
# ----------------------
class JesterMultiFrameDataset(Dataset):
    """
    CSV 每行: "video_id;label_name"
    帧目录: FRAME_ROOT/<video_id>/*.jpg
    返回: Tensor [T, C, H, W] 以及标签
    """

    def __init__(
        self,
        csv_path: str,
        frame_root: str,
        label2idx: dict,
        transform: Optional[object] = None,
        max_samples: Optional[int] = None,
        frames_per_clip: int = 8,
        is_train: bool = True,
    ):
        self.frame_root = frame_root
        self.label2idx = label2idx
        self.transform = transform
        self.is_train = is_train
        self.frames_per_clip = frames_per_clip

        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] == 1:
            df = df[0].str.split(";", expand=True)
        df.columns = ["video_id", "label_name"]

        df["video_id"] = df["video_id"].astype(str).str.strip()
        df["label_name"] = df["label_name"].astype(str).str.strip()

        # 实际有哪些视频文件夹
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

    def _get_frame_files(self, video_dir: str) -> List[str]:
        files = sorted(
            f for f in os.listdir(video_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not files:
            raise RuntimeError(f"No frames found in {video_dir}")
        return [os.path.join(video_dir, f) for f in files]

    # def _sample_indices(self, num_frames: int) -> List[int]:
    #     T = self.frames_per_clip
    #     if num_frames >= T:
    #         # 均匀采样 T 帧
    #         indices = np.linspace(0, num_frames - 1, T, dtype=int).tolist()
    #     else:
    #         # 帧数不够: 重复某些帧填满 T
    #         base = np.arange(num_frames)
    #         reps = int(np.ceil(T / num_frames))
    #         tiled = np.tile(base, reps)
    #         indices = tiled[:T].tolist()
    #     return indices

    def _sample_indices(self, num_frames: int) -> List[int]:
        """
        从一个视频中采样 self.frames_per_clip 帧的索引。
        - 训练集: 随机采样（时间顺序排序），起到 temporal augmentation 作用
        - 验证/测试集: 均匀采样，结果可复现、稳定
        """
        T = self.frames_per_clip

        if self.is_train:
            # ---------- Train mode: 随机采样 ----------
            if num_frames >= T:
                # 从所有帧中随机选 T 帧，再按时间顺序排序
                indices = np.sort(
                    np.random.choice(num_frames, size=T, replace=False)
                ).tolist()
            else:
                # 帧数不够: 允许重复采样，仍然随机
                indices = np.sort(
                    np.random.choice(num_frames, size=T, replace=True)
                ).tolist()
        else:
            # ---------- Val / Test mode: 均匀采样 ----------
            if num_frames >= T:
                # 均匀采 T 帧
                indices = np.linspace(0, num_frames - 1, T, dtype=int).tolist()
            else:
                # 帧数不够: 重复帧填满 T
                base = np.arange(num_frames)
                reps = int(np.ceil(T / num_frames))
                tiled = np.tile(base, reps)
                indices = tiled[:T].tolist()

        return indices

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        video_id = row["video_id"]
        label_name = row["label_name"]

        video_dir = self._get_video_dir(video_id)
        frame_files = self._get_frame_files(video_dir)
        indices = self._sample_indices(len(frame_files))

        frames = []
        for i in indices:
            img = cv2.imread(frame_files[i])
            if img is None:
                raise RuntimeError(f"Cannot read frame {frame_files[i]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                img_t = self.transform(img)  # [C,H,W]
            else:
                img_t = T.ToTensor()(img)
            frames.append(img_t)

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]

        label_idx = self.label2idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)

        return frames, label


# ----------------------
# Transforms
# ----------------------
def get_transforms(is_train: bool = True):
    if is_train:
        ops = [
            T.ToPILImage(),
            T.RandomResizedCrop(CONFIG.IMG_SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
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
# Time-series Model: ResNet18 backbone + GRU
# ----------------------
class GestureTimeSeriesNet(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256, num_layers: int = 1):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        # 冻结 backbone 的大部分参数，只微调最后一层 block
        for name, param in backbone.named_parameters():
            param.requires_grad = False

        for name, param in backbone.layer3.named_parameters():
            param.requires_grad = True
        for name, param in backbone.layer4.named_parameters():
            param.requires_grad = True


        self.backbone = backbone

        self.rnn = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=0.6)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.backbone(x)          # [B*T, F]
        feats = feats.view(B, T, -1)      # [B, T, F]

        out, _ = self.rnn(feats)          # [B, T, 2H]

        # 使用所有时间步的平均，而不是仅最后一步
        time_pooled = out.mean(dim=1)     # [B, 2H]

        time_pooled = self.dropout(time_pooled)
        logits = self.classifier(time_pooled)
        return logits


# ----------------------
# Train / Eval
# ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch_idx, (frames, labels) in enumerate(loader):
        if batch_idx == 0:
            print("  [DEBUG] first train batch loaded")
        if (batch_idx + 1) % 50 == 0:
            print(f"  [DEBUG] val batch {batch_idx+1}")

        frames = frames.to(device)   # [B, T, C, H, W]
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += frames.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch_idx, (frames, labels) in enumerate(loader):
        if batch_idx == 0:
            print("  [DEBUG] first val batch loaded")
        if (batch_idx + 1) % 50 == 0:
            print(f"  [DEBUG] val batch {batch_idx+1}")
        frames = frames.to(device)
        labels = labels.to(device)

        logits = model(frames)
        loss = criterion(logits, labels)

        total_loss += loss.item() * frames.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += frames.size(0)
    print('evaluate are done')
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
# Main with early stopping + ReduceLROnPlateau
# ----------------------
def main():
    label2idx, idx2label = load_label_mapping(CONFIG.LABELS_CSV)

    train_dataset = JesterMultiFrameDataset(
        csv_path=CONFIG.TRAIN_CSV,
        frame_root=CONFIG.FRAME_ROOT,
        label2idx=label2idx,
        transform=get_transforms(is_train=True),
        max_samples=CONFIG.MAX_TRAIN_SAMPLES,
        frames_per_clip=CONFIG.FRAMES_PER_CLIP,
        is_train=True,
    )
    val_dataset = JesterMultiFrameDataset(
        csv_path=CONFIG.VAL_CSV,
        frame_root=CONFIG.FRAME_ROOT,
        label2idx=label2idx,
        transform=get_transforms(is_train=False),
        max_samples=None,
        frames_per_clip=CONFIG.FRAMES_PER_CLIP,
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
    print(f'device is {CONFIG.DEVICE}')
    model = GestureTimeSeriesNet(CONFIG.NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG.LABEL_SMOOTHING)
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": CONFIG.LR * 0.3},  # backbone 用更小 LR
            {"params": head_params,   "lr": CONFIG.LR},          # GRU + classifier 用原 LR
        ],
        weight_decay=CONFIG.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=True,
    )


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

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{CONFIG.NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        improve_eps = 1e-4  # 很小的阈值，避免浮点数问题
        if val_acc > best_val_acc + improve_eps:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "improve_best.pth"))
            print(f"  -> New best val acc {best_val_acc:.4f}, model saved.")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= CONFIG.EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    plot_history(history, prefix="Improve time series", out_dir=BASE_DIR)
    print("Saved curves to 'improve_time_series_loss.png' and 'improve_time_series_acc.png'.")
    print(f"Best val accuracy during training: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
