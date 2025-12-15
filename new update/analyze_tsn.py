"""
analyze_tsn.py

在验证集上分析 TSN 提升模型：
- 载入 improve_tsn.py 中的数据集与模型结构
- 加载训练好的权重 improve_best_tsn.pth
- 计算整体准确率、分类报告
- 生成并保存：
    1) TSN 混淆矩阵（行归一化）
    2) TSN 每类准确率条形图
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------
# 从 improve_tsn.py 导入训练时用到的组件
# -----------------------
from improve_tsn import (
    CONFIG,             # 和训练时相同的配置
    get_transforms,
    load_label_mapping,
)

# 数据集类和模型类的名字你可能略有不同，这里做一个兼容：
try:
    from improve_tsn import JesterTSNDataset as TSNDataset
except ImportError:
    from improve_tsn import JesterMultiFrameDataset as TSNDataset

try:
    from improve_tsn import GestureTSNNet as TSNModel
except ImportError:
    from improve_tsn import GestureTimeSeriesNet as TSNModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "improve_best_tsn.pth"   # TSN 最佳模型的权重文件名


# -----------------------
# 构建验证集 DataLoader
# -----------------------
def build_val_loader():
    label2idx, idx2label = load_label_mapping(CONFIG.LABELS_CSV)

    val_dataset = TSNDataset(
        csv_path=CONFIG.VAL_CSV,
        frame_root=CONFIG.FRAME_ROOT,
        label2idx=label2idx,
        transform=get_transforms(is_train=False),
        max_samples=None,                        # 验证集用全部
        frames_per_clip=getattr(CONFIG, "FRAMES_PER_CLIP", 8),
        is_train=False,
    )

    # 为了避免 Windows 下多进程 dataloader 的各种 pickling 问题，这里 num_workers=0
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return val_dataset, val_loader, idx2label


# -----------------------
# 载入模型
# -----------------------
def load_model(checkpoint_path):
    model = TSNModel(CONFIG.NUM_CLASSES)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# -----------------------
# 收集预测结果（带简单进度提示）
# -----------------------
@torch.no_grad()
def collect_predictions(model, loader):
    all_preds, all_labels = [], []
    total_batches = len(loader)

    for i, (frames, labels) in enumerate(loader):
        # TSN 输入形状: [B, T, C, H, W]
        frames = frames.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(frames)
        preds = logits.argmax(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        if (i + 1) % 50 == 0 or (i + 1) == total_batches:
            print(f"[{i+1}/{total_batches}] batches processed")

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    return all_labels, all_preds


# -----------------------
# 可视化：混淆矩阵
# -----------------------
def plot_confusion_matrix(y_true, y_pred, idx2label, prefix="improve_tsn"):
    num_classes = len(idx2label)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 7))
    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    plt.title(f"{prefix} confusion matrix (row-normalized)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, idx2label, rotation=90, fontsize=5)
    plt.yticks(tick_marks, idx2label, fontsize=5)

    plt.tight_layout()
    out_path = f"{prefix}_confusion_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# -----------------------
# 可视化：每类准确率
# -----------------------
def plot_per_class_accuracy(y_true, y_pred, idx2label, prefix="improve_tsn"):
    num_classes = len(idx2label)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    order = np.argsort(per_class_acc)
    sorted_acc = per_class_acc[order]
    sorted_labels = np.array(idx2label)[order]

    plt.figure(figsize=(8, 7))
    plt.barh(sorted_labels, sorted_acc)
    plt.xlabel("Per-class accuracy")
    plt.ylabel("Class")
    plt.title(f"{prefix} per-class accuracy")
    plt.tight_layout()
    out_path = f"{prefix}_per_class_accuracy.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    print("Top worst 5")
    for lbl, acc in zip(sorted_labels[:5], sorted_acc[:5]):
        print(f"  {lbl:30s}: {acc:.3f}")
    print("Top best 5：")
    for lbl, acc in zip(sorted_labels[-5:], sorted_acc[-5:]):
        print(f"  {lbl:30s}: {acc:.3f}")


# -----------------------
# 主流程
# -----------------------
if __name__ == "__main__":
    print("Building validation loader ...")
    val_dataset, val_loader, idx2label = build_val_loader()

    print(f"Validation samples: {len(val_dataset)}")
    print("Loading TSN model from", CHECKPOINT)
    model = load_model(CHECKPOINT)

    print("Collecting predictions on validation set ...")
    y_true, y_pred = collect_predictions(model, val_loader)

  
    np.save("improve_tsn_y_true_val.npy", y_true)
    np.save("improve_tsn_y_pred_val.npy", y_pred)

    overall_acc = (y_true == y_pred).mean()
    print(f"Overall validation accuracy: {overall_acc:.4f}")

    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=idx2label,
            digits=3,
        )
    )

    plot_confusion_matrix(y_true, y_pred, idx2label, prefix="improve1_tsn")
    plot_per_class_accuracy(y_true, y_pred, idx2label, prefix="improve1_tsn")
