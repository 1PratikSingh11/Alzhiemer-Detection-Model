"""
Utility functions for Alzheimer's Disease Detection Model (ResNet-18 + SHAP).

Includes:
- MRI image transformations (Resize to 248x496 -> CenterCrop to 224x224 -> ImageNet Norm)
- Dataset loaders and stratified 80/20 train/validation split
- Class-weighted Cross-Entropy loss computation to counteract severe class imbalance
- Comprehensive metrics evaluation (Accuracy, Macro Precision, Recall, F1, Mean Confidence)
- Plotting utilities for training curves, confusion matrix, and ROC curves
"""

import os
import json
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import CLASS_NAMES

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Returns image transformation pipeline according to the project specifications:
    1. Resize to 248x496
    2. CenterCrop to 224x224
    3. Normalize using ImageNet statistics
    
    If augment=True, applies subtle MRI-appropriate augmentations (slight rotation,
    horizontal flip, affine transformation) to enhance generalization.
    """
    transform_list = []

    # Target resize as documented in final report (pages 1-4)
    transform_list.append(transforms.Resize((248, 496)))

    if augment:
        transform_list.extend([
            transforms.RandomRotation(degrees=7),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        ])

    transform_list.extend([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        # Handle 1-channel grayscale conversion to 3-channel if necessary
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x[:3, :, :]),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transforms.Compose(transform_list)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverses ImageNet normalization on a (C, H, W) tensor and returns
    a (H, W, C) numpy array in range [0, 1] suitable for plotting and SHAP overlays.
    """
    img = tensor.clone().detach().cpu().numpy()
    if img.ndim == 3:
        # Transpose from (C, H, W) to (H, W, C)
        img = np.transpose(img, (1, 2, 0))
    
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return img


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """
    Calculates balanced class weights inversely proportional to class frequencies:
    w_c = N / (C * N_c)
    
    This handles the severe OASIS dataset imbalance (e.g. Non-Demented >> Moderate Dementia).
    """
    targets = []
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "indices") and hasattr(dataset.dataset, "targets"):
        # Subset dataset
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        # Fallback iteration
        for _, label in dataset:
            targets.append(label)

    class_counts = np.bincount(targets, minlength=len(CLASS_NAMES))
    total_samples = len(targets)
    num_classes = len(CLASS_NAMES)

    # Avoid zero division
    weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(np.float32))
    # Normalize so sum of weights equals num_classes
    weights = weights / np.mean(weights)
    
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_loss(dataset: Dataset, device: torch.device) -> nn.CrossEntropyLoss:
    """Returns nn.CrossEntropyLoss configured with inverse frequency weights."""
    weights = compute_class_weights(dataset).to(device)
    print(f"[Loss Config] Class Weights: { {c: round(w.item(), 4) for c, w in zip(CLASS_NAMES, weights)} }")
    return nn.CrossEntropyLoss(weight=weights)


def load_dataset_splits(
    data_dir: str,
    train_val_split: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """
    Loads dataset from directory, applies 80/20 train/validation split,
    and returns DataLoaders and Datasets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    # Check if train and val folders exist separately
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    else:
        # Unified directory: load and split 80/20
        full_dataset_train = datasets.ImageFolder(data_dir, transform=train_transform)
        full_dataset_val = datasets.ImageFolder(data_dir, transform=val_transform)

        total_samples = len(full_dataset_train)
        indices = list(range(total_samples))
        split_idx = int(np.floor(train_val_split * total_samples))

        np.random.shuffle(indices)
        train_indices, val_indices = indices[:split_idx], indices[split_idx:]

        train_dataset = Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_val, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, train_dataset, val_dataset


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes all standard classification metrics:
    - Overall Accuracy
    - Macro Precision, Recall, F1-Score
    - Mean Prediction Confidence
    - Confusion Matrix
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_probs is not None:
        max_confidences = np.max(y_probs, axis=1)
        metrics["mean_confidence"] = float(np.mean(max_confidences))

    return metrics


def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: str = "outputs_tcc_resnet18"
) -> str:
    """Plots training and validation loss and accuracy curves."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Training Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    axes[0].set_title("Training and Validation Loss Curve", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Weighted Cross-Entropy Loss", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy / F1 Curve
    axes[1].plot(epochs, history["val_accuracy"], "g-o", label="Validation Accuracy", linewidth=2)
    if "val_f1" in history:
        axes[1].plot(epochs, history["val_f1"], "m--^", label="Validation Macro F1", linewidth=2)
    axes[1].set_title("Validation Accuracy & Macro F1-Score", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].set_ylim([0.0, 1.05])
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_validation_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    output_dir: str = "outputs_tcc_resnet18",
    normalize: bool = False
) -> str:
    """Renders and saves high-resolution confusion matrix heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"
        title = "Normalized Confusion Matrix (OASIS Validation)"
        filename = "confusion_matrix_normalized.png"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix (OASIS Validation Set)"
        filename = "confusion_matrix_raw.png"

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1.0,
        linecolor="white",
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Disease Severity", fontsize=12, labelpad=10)
    plt.ylabel("True Clinical Label", fontsize=12, labelpad=10)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_path


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    output_dir: str = "outputs_tcc_resnet18"
) -> str:
    """Plots One-vs-Rest ROC curves and calculates per-class AUC scores."""
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(class_names)
    
    plt.figure(figsize=(9, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i in range(n_classes):
        # Binary target for class i vs rest
        y_true_bin = (y_true == i).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            lw=2.5,
            label=f"{class_names[i]} (AUC = {roc_auc:.4f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance Level (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("Multi-Class One-vs-Rest ROC Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    plot_path = os.path.join(output_dir, "multiclass_roc_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_path
