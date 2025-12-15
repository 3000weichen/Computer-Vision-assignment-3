# analyze_improve1.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 1) 从 Improve-1 脚本中导入配置、数据集和模型
from improve_1 import (
    CONFIG,
    JesterFrameFolderDataset,
    get_transforms,
    load_label_mapping,
    GestureImprovedNet,
)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "improved_best.pth"   # <-- 使用 Improve-1 训练出的最优模型
PREFIX     = "improve1"           # 输出图片/文件名前缀

# --------------------------------------------------
# 构造验证集 DataLoader
# --------------------------------------------------
def build_val_loader():
    label2idx, idx2label = load_label_mapping(CONFIG.LABELS_CSV)

    val_dataset = JesterFrameFolderDataset(
        csv_path=CONFIG.VAL_CSV,
        frame_root=CONFIG.FRAME_ROOT,
        label2idx=label2idx,
        transform=get_transforms(is_train=False),
        max_samples=None,      # 验证集全部使用
        is_train=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=False,
        num_workers=0,         # 分析时设为 0，避免 Windows 多进程问题
        pin_memory=True,
    )
    return val_dataset, val_loader, idx2label

# --------------------------------------------------
# 加载模型权重
# --------------------------------------------------
def load_model(checkpoint_path):
    model = GestureImprovedNet(CONFIG.NUM_CLASSES)   # <-- 用改进后的网络结构
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# --------------------------------------------------
# 收集预测结果
# --------------------------------------------------
def collect_predictions(model, loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            # 简单的进度提示，方便确认程序在跑
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(loader)}] batches processed")

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    return all_labels, all_preds

# --------------------------------------------------
# 画混淆矩阵（行归一化）
# --------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, idx2label, prefix=PREFIX):
    num_classes = len(idx2label)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 7))
    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    plt.title(f"{prefix} confusion matrix (row-normalized)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    ticks = np.arange(num_classes)
    plt.xticks(ticks, idx2label, rotation=90, fontsize=5)
    plt.yticks(ticks, idx2label, fontsize=5)

    plt.tight_layout()
    plt.savefig(f"{prefix}_confusion_matrix.png",
                dpi=300, bbox_inches="tight")
    plt.close()

# --------------------------------------------------
# 画 per-class accuracy 柱状图
# --------------------------------------------------
def plot_per_class_accuracy(y_true, y_pred, idx2label, prefix=PREFIX):
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
    plt.savefig(f"{prefix}_per_class_accuracy.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    print("Top 5 worst：")
    for lbl, acc in zip(sorted_labels[:5], sorted_acc[:5]):
        print(f"  {lbl:30s}: {acc:.3f}")

    print("Top 5 best：")
    for lbl, acc in zip(sorted_labels[-5:], sorted_acc[-5:]):
        print(f"  {lbl:30s}: {acc:.3f}")

# --------------------------------------------------
# 主流程
# --------------------------------------------------
if __name__ == "__main__":
    val_dataset, val_loader, idx2label = build_val_loader()
    model = load_model(CHECKPOINT)
    y_true, y_pred = collect_predictions(model, val_loader)

    np.save(f"{PREFIX}_y_true_val.npy", y_true)
    np.save(f"{PREFIX}_y_pred_val.npy", y_pred)

    print("Total acc:", (y_true == y_pred).mean())
    print(
        classification_report(
            y_true, y_pred,
            target_names=idx2label,
            digits=3,
        )
    )

    plot_confusion_matrix(y_true, y_pred, idx2label, prefix=PREFIX)
    plot_per_class_accuracy(y_true, y_pred, idx2label, prefix=PREFIX)
