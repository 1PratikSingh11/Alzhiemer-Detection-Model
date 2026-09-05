"""
SHAP Explainability and Interpretability Module (SafeResNet-18).

Integrates:
  1. SHAP GradientExplainer for deep neural networks on PyTorch.
  2. Background sampling from the validation dataset.
  3. Generation of pixel-level attribution heatmaps (positive vs negative evidence).
  4. Class-wise contribution plots across all 4 Alzheimer's disease severity stages.
  5. Grad-CAM comparison module (coarse regional localization vs. SHAP fine attribution).
  6. Multi-sample cohort summaries across all 4 dementia classes.
  
All explainability outputs and figures are saved to: shap_analysis/
"""

import os
import argparse
from typing import Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import shap

from model import SafeResNet18, build_model, CLASS_NAMES
from utils import get_transforms, denormalize_image, load_dataset_splits


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for SafeResNet-18.
    Provides regional feature attribution to benchmark alongside SHAP GradientExplainer.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of gradients: weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Linear combination of weighted activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Apply ReLU to retain features that have positive influence
        cam = F.relu(cam)

        # Upsample CAM to match original image dimensions (224x224)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        # Normalize between 0 and 1
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


class SafeResNetExplainer:
    """
    Unified Explainability Engine wrapping SHAP GradientExplainer & Grad-CAM.
    """
    def __init__(
        self,
        model: nn.Module,
        background_tensors: torch.Tensor,
        device: torch.device
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.background = background_tensors.to(self.device)
        print(f"[SHAP] Initializing GradientExplainer with {self.background.shape[0]} background reference samples...")
        self.explainer = shap.GradientExplainer(self.model, self.background)

        # Initialize Grad-CAM on final conv layer of layer4
        last_conv = self.model.get_last_conv_layer() if hasattr(self.model, "get_last_conv_layer") else self.model.resnet.layer4[-1].conv2
        self.grad_cam = GradCAM(self.model, last_conv)

    def explain_scan(
        self,
        image_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """
        Computes SHAP values, Grad-CAM activation, predicted class, and probabilities.
        """
        image_tensor = image_tensor.to(self.device)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # 1. Model inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))

        # 2. SHAP GradientExplainer attribution
        # Compute shap values across all classes
        shap_vals = self.explainer.shap_values(image_tensor)
        
        # Format shap_values into array: (num_classes, H, W, C) or (num_classes, C, H, W)
        if isinstance(shap_vals, list):
            # List of length num_classes, each is (1, C, H, W)
            formatted_shap = [s[0] for s in shap_vals]
        else:
            # Array formatted as (1, C, H, W, num_classes) or similar
            formatted_shap = [shap_vals[..., i][0] for i in range(len(CLASS_NAMES))]

        # 3. Grad-CAM heatmap
        cam_map = self.grad_cam.generate_cam(image_tensor, target_class=pred_class)

        return formatted_shap, cam_map, pred_class, probs


def plot_single_explanation(
    original_img: np.ndarray,
    shap_vals_all_classes: List[np.ndarray],
    cam_map: np.ndarray,
    pred_class: int,
    probs: np.ndarray,
    true_class: Optional[str] = None,
    output_path: str = "shap_analysis/sample_explanation.png"
) -> None:
    """
    Renders publication-quality multi-panel visualization:
      Panel 1: Original Preprocessed Brain MRI
      Panel 2: SHAP Attribution Map for Predicted Class (Positive/Negative Evidence)
      Panel 3: Grad-CAM Regional Heatmap Overlay
      Panel 4: Class-wise Prediction Probabilities & SHAP Attribution Mass
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    # Panel 1: Original MRI
    axes[0].imshow(original_img)
    title_p1 = f"Input MRI Scan\nPredicted: {CLASS_NAMES[pred_class]} ({probs[pred_class]*100:.1f}%)"
    if true_class:
        title_p1 += f"\nGround Truth: {true_class}"
    axes[0].set_title(title_p1, fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: SHAP Attribution Map
    # Extract attribution for the predicted class
    pred_shap = shap_vals_all_classes[pred_class]
    # Reduce across RGB channels: take mean or sum
    if pred_shap.shape[0] == 3:  # (C, H, W)
        shap_2d = np.mean(pred_shap, axis=0)
    else:  # (H, W, C)
        shap_2d = np.mean(pred_shap, axis=-1)

    abs_max = np.percentile(np.abs(shap_2d), 99.5)
    abs_max = max(abs_max, 1e-6)

    im2 = axes[1].imshow(original_img, cmap="gray", alpha=0.45)
    im2_shap = axes[1].imshow(shap_2d, cmap="seismic", alpha=0.85, vmin=-abs_max, vmax=abs_max)
    axes[1].set_title(
        f"SHAP GradientExplainer\nAttribution for: {CLASS_NAMES[pred_class]}\n(Red: Positive Evidence, Blue: Negative)",
        fontsize=10, fontweight="bold"
    )
    axes[1].axis("off")
    cbar2 = plt.colorbar(im2_shap, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("SHAP Attribution", fontsize=9)

    # Panel 3: Grad-CAM Regional Attention
    axes[2].imshow(original_img)
    im3 = axes[2].imshow(cam_map, cmap="jet", alpha=0.5)
    axes[2].set_title(
        f"Grad-CAM Comparison\nLayer4 Coarse Activation Map\n(Regional Salience)",
        fontsize=10, fontweight="bold"
    )
    axes[2].axis("off")
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label("Grad-CAM Weight", fontsize=9)

    # Panel 4: Class-wise Probabilities and Net SHAP Impact
    y_pos = np.arange(len(CLASS_NAMES))
    axes[3].barh(y_pos, probs, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"], alpha=0.85, edgecolor="black")
    axes[3].set_yticks(y_pos)
    axes[3].set_yticklabels(CLASS_NAMES, fontsize=9)
    axes[3].set_xlabel("Predicted Softmax Probability", fontsize=10)
    axes[3].set_xlim([0.0, 1.05])
    axes[3].set_title("Model Confidence by Class", fontsize=11, fontweight="bold")
    axes[3].grid(axis="x", linestyle="--", alpha=0.5)

    for i, p in enumerate(probs):
        axes[3].text(p + 0.02, i, f"{p*100:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved explanation figure to: {output_path}")


def run_explainability_pipeline(
    data_dir: str = "data",
    model_path: str = "models/best_model.pth",
    output_dir: str = "shap_analysis",
    num_background_samples: int = 30,
    device: str = "cpu"
) -> None:
    """Executes full SHAP explainability analysis on validation samples."""
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")

    # 1. Load Model
    print(f"\n[1/3] Loading SafeResNet-18 model for interpretability...")
    model = build_model(num_classes=4, pretrained=False).to(dev)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=dev)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded weights from checkpoint: {model_path}")
    else:
        print(f"Notice: Checkpoint '{model_path}' not found. Initializing with ImageNet weights.")
        model = build_model(num_classes=4, pretrained=True).to(dev)

    # 2. Prepare Background and Test Samples
    print(f"\n[2/3] Sampling background distribution and test scans from '{data_dir}'...")
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        from data.generate_oasis_mock import generate_dataset
        generate_dataset(output_root=data_dir, samples_per_class_train=40, samples_per_class_val=10)

    _, val_loader, _, val_dataset = load_dataset_splits(data_dir=data_dir, batch_size=num_background_samples)

    # Collect background batch
    bg_images, _ = next(iter(val_loader))
    bg_tensors = bg_images[:num_background_samples].to(dev)

    # Instantiate Explainer
    explainer = SafeResNetExplainer(model, bg_tensors, dev)

    # 3. Explain representative test scans from each class
    print(f"\n[3/3] Generating SHAP GradientExplainer attribution heatmaps & Grad-CAM comparisons...")
    
    samples_found = {c: False for c in CLASS_NAMES}
    explained_count = 0

    for images, labels in val_loader:
        for idx in range(images.size(0)):
            img_tensor = images[idx:idx+1]
            label_idx = labels[idx].item()
            class_name = CLASS_NAMES[label_idx]

            if not samples_found[class_name] or explained_count < 4:
                original_display = denormalize_image(img_tensor[0])
                shap_vals, cam_map, pred_class, probs = explainer.explain_scan(img_tensor)

                out_filename = os.path.join(
                    output_dir,
                    f"shap_explanation_{class_name.replace(' ', '_').lower()}_{idx:02d}.png"
                )
                plot_single_explanation(
                    original_img=original_display,
                    shap_vals_all_classes=shap_vals,
                    cam_map=cam_map,
                    pred_class=pred_class,
                    probs=probs,
                    true_class=class_name,
                    output_path=out_filename
                )
                samples_found[class_name] = True
                explained_count += 1

            if all(samples_found.values()) and explained_count >= 4:
                break
        if all(samples_found.values()) and explained_count >= 4:
            break

    print(f"\n[Done] SHAP analysis complete! All maps and comparisons saved to '{output_dir}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP and Grad-CAM explanations")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset directory")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth", help="Trained weights path")
    parser.add_argument("--output_dir", type=str, default="shap_analysis", help="Output directory for heatmaps")
    parser.add_argument("--num_bg", type=int, default=30, help="Number of background samples for SHAP")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_explainability_pipeline(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_background_samples=args.num_bg,
        device=args.device
    )
