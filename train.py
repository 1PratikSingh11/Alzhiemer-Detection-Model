"""
Training Pipeline for Alzheimer's Disease Detection (SafeResNet-18).

Aligned with the methodology and hyperparameter specifications:
  - Architecture: SafeResNet18 (inplace=False ReLUs, 512 -> 4 classifier)
  - Loss Function: Weighted Cross-Entropy (handling class imbalance)
  - Optimizer: Adam (lr = 1e-4)
  - Learning Rate Scheduler: StepLR (step_size=5, gamma=0.1)
  - Input Resolution: 224 x 224 (Resize to 248x496 -> CenterCrop to 224x224 -> ImageNet Norm)
  - Batch Size: 32
  - Target Evaluation Metrics: Accuracy, Macro Precision, Macro Recall, Macro F1-score
  - Logs, curves, and confusion matrix saved to: outputs_tcc_resnet18/
  - Best model checkpoint saved to: models/best_model.pth
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from model import SafeResNet18, build_model, CLASS_NAMES
from utils import (
    load_dataset_splits,
    get_weighted_loss,
    compute_metrics,
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curves
)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Trains the model for one epoch and returns average training loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(total_samples, 1)
    return epoch_loss


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluates the model on validation set and computes full metrics suite."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="[Validation]", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.extend(preds.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_probs)
    metrics["val_loss"] = running_loss / max(total_samples, 1)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_probs"] = y_probs

    return metrics


def run_training(args: argparse.Namespace) -> None:
    """Main training routine."""
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"============================================================")
    print(f"Alzheimer's Disease Detection Model: SafeResNet-18 + SHAP")
    print(f"Device: {device} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr} | StepLR: step_size={args.step_size}, gamma={args.gamma}")
    print(f"============================================================")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    # 1. Load Data
    print(f"\n[1/5] Loading MRI dataset from '{args.data_dir}'...")
    if not os.path.exists(args.data_dir) or not os.listdir(args.data_dir):
        print(f"Warning: Dataset not found in '{args.data_dir}'.")
        print("Generating realistic synthetic OASIS sample dataset for demonstration...")
        from data.generate_oasis_mock import generate_dataset
        generate_dataset(output_root=args.data_dir, samples_per_class_train=40, samples_per_class_val=10)

    train_loader, val_loader, train_dataset, val_dataset = load_dataset_splits(
        data_dir=args.data_dir,
        train_val_split=args.train_val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f"Loaded {len(train_dataset)} training scans and {len(val_dataset)} validation scans.")

    # 2. Build SafeResNet-18 Model
    print(f"\n[2/5] Initializing SafeResNet-18 architecture (inplace=False ReLUs)...")
    model = build_model(
        num_classes=len(CLASS_NAMES),
        pretrained=args.pretrained,
        dropout_rate=args.dropout
    ).to(device)

    # 3. Setup Weighted Loss & Optimizer
    print(f"\n[3/5] Configuring Weighted Cross-Entropy Loss and Adam Optimizer...")
    criterion = get_weighted_loss(train_dataset, device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # 4. Training Loop
    print(f"\n[4/5] Beginning Model Training...")
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
    }

    best_f1 = -1.0
    best_metrics: Dict[str, Any] = {}
    best_checkpoint_path = os.path.join(args.models_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["macro_precision"])
        history["val_recall"].append(val_metrics["macro_recall"])
        history["val_f1"].append(val_metrics["macro_f1"])

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Save Best Model Checkpoint
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_f1": best_f1,
                "accuracy": val_metrics["accuracy"],
                "class_names": CLASS_NAMES
            }, best_checkpoint_path)
            print(f"  --> Best model checkpoint saved to: {best_checkpoint_path} (Macro F1: {best_f1:.4f})")

    # Save Final Model Checkpoint
    final_checkpoint_path = os.path.join(args.models_dir, "final_model.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")

    # 5. Export Logs and Visualizations
    print(f"\n[5/5] Exporting Experiment Logs and Visualizations to '{args.output_dir}'...")

    # Plot loss and accuracy curves
    plot_training_curves(history, output_dir=args.output_dir)

    # Plot confusion matrices
    if "confusion_matrix" in best_metrics:
        cm = np.array(best_metrics["confusion_matrix"])
        plot_confusion_matrix(cm, class_names=CLASS_NAMES, output_dir=args.output_dir, normalize=False)
        plot_confusion_matrix(cm, class_names=CLASS_NAMES, output_dir=args.output_dir, normalize=True)

    # Plot ROC curves
    if "y_true" in best_metrics and "y_probs" in best_metrics:
        plot_roc_curves(best_metrics["y_true"], best_metrics["y_probs"], class_names=CLASS_NAMES, output_dir=args.output_dir)

    # Save JSON summary
    summary_report = {
        "final_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else None,
        "best_macro_f1": best_f1,
        "best_accuracy": best_metrics.get("accuracy"),
        "best_macro_precision": best_metrics.get("macro_precision"),
        "best_macro_recall": best_metrics.get("macro_recall"),
        "best_val_loss": best_metrics.get("val_loss"),
        "mean_confidence": best_metrics.get("mean_confidence"),
        "reported_benchmark_reference": {
            "accuracy": 0.9890,
            "macro_precision": 0.9787,
            "macro_recall": 0.9954,
            "macro_f1": 0.9868,
            "mean_confidence": 0.9882,
            "validation_loss": 0.0170,
            "training_loss": 0.0099,
            "confusion_matrix": {
                "Non-Demented": 13271,
                "Very Mild Dementia": 2753,
                "Mild Dementia": 993,
                "Moderate Dementia": 80
            }
        },
        "history": history
    }

    report_file = os.path.join(args.output_dir, "experiment_results.json")
    with open(report_file, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Experiment results saved to: {report_file}")
    print("\nTraining workflow successfully completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SafeResNet-18 on MRI Scans")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate (default: 1e-4)")
    parser.add_argument("--step_size", type=int, default=5, help="StepLR step size (default: 5)")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma decay (default: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Optimizer weight decay")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate before classifier")
    parser.add_argument("--train_val_split", type=float, default=0.8, help="Train/val split ratio (default: 0.8)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num workers")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save model weights")
    parser.add_argument("--output_dir", type=str, default="outputs_tcc_resnet18", help="Output directory for plots/logs")

    args = parser.parse_args()
    run_training(args)
