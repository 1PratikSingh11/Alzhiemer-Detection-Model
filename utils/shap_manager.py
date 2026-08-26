# utils/shap_manager.py
"""
SHAPInteractiveManager with graceful fallback.
If shap package is available it uses shap.GradientExplainer.
If shap is not available or fails, it falls back to a gradient-saliency map.
Saves a PNG in the provided output_dir and returns metadata.
"""

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

class SHAPInteractiveManager:
    def __init__(self, model, device, class_names, output_dir):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.background_data = None
        self.explainer = None
        # Try to import shap lazily
        try:
            import shap
            self.shap = shap
            self.shap_available = True
        except Exception:
            self.shap = None
            self.shap_available = False

    def prepare_shap_data_from_single_image(self, tensor_batch):
        """Use single image as background (useful for demo)."""
        if isinstance(tensor_batch, torch.Tensor):
            self.background_data = tensor_batch.to(self.device)
        else:
            self.background_data = torch.tensor(tensor_batch).float().to(self.device)

    def create_shap_explainer(self):
        """Create shap explainer if shap is available."""
        if not self.shap_available:
            # nothing to create; fallback will be used
            return
        try:
            # shap expects a function that returns model output probabilities as numpy array
            def predict(x):
                # x is numpy array (batch, C, H, W)
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float().to(self.device)
                with torch.no_grad():
                    out = self.model(x)
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                return probs
            # create GradientExplainer with background
            self.explainer = self.shap.GradientExplainer(predict, self.background_data)
        except Exception as e:
            # If shap fails, mark unavailable
            self.shap_available = False
            self.explainer = None

    def analyze_single_image(self, image_tensor, image_path, save_prefix="shap"):
        """
        Run SHAP if available, otherwise compute gradient-saliency fallback.
        Returns dict: {'predicted_class', 'confidence', 'shap_file'}
        """
        img_batch = image_tensor.unsqueeze(0).to(self.device)

        # model outputs
        with torch.no_grad():
            out = self.model(img_batch)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)
            pred_class = int(pred.item())
            conf_val = float(conf.item())

        if self.shap_available and self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(img_batch.cpu().numpy())
                # shap_values shape: list[num_classes] of arrays (batch, C, H, W)
                # choose predicted class heatmap
                if isinstance(shap_values, list) and len(shap_values) > pred_class:
                    heat = np.mean(np.abs(shap_values[pred_class][0]), axis=0)
                else:
                    # fallback if unexpected format
                    heat = np.mean(np.abs(shap_values[0][0]), axis=0)
            except Exception:
                # fallback to gradient saliency if shap computation fails
                heat = self._grad_saliency(image_tensor, pred_class)
        else:
            # fallback: gradient saliency
            heat = self._grad_saliency(image_tensor, pred_class)

        # normalize heatmap to 0-1
        heat = heat - np.min(heat)
        if np.max(heat) > 0:
            heat = heat / np.max(heat)
        heat_img = (heat * 255).astype(np.uint8)

        # save png
        save_path = os.path.join(self.output_dir, f"{save_prefix}_{self.class_names[pred_class].replace(' ', '_')}_{conf_val:.3f}.png")
        plt.figure(figsize=(5,5))
        plt.imshow(heat_img, cmap="hot")
        plt.axis("off")
        plt.title(f"SHAP / Saliency - {self.class_names[pred_class]} ({conf_val:.3f})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        return {"predicted_class": pred_class, "confidence": conf_val, "shap_file": save_path}

    def _grad_saliency(self, image_tensor, target_class):
        """Simple gradient-based saliency map (fallback)."""
        self.model.zero_grad()
        img = image_tensor.unsqueeze(0).to(self.device)
        img.requires_grad_()
        out = self.model(img)
        score = out[0, target_class]
        score.backward(retain_graph=True)
        grad = img.grad.detach().cpu().numpy()[0]  # C,H,W
        # aggregate absolute gradient over channels
        sal = np.mean(np.abs(grad), axis=0)
        # size may be 224x224 depending on transform
        return sal
