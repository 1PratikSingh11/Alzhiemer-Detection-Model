"""
Inference CLI Tool for Alzheimer's Disease MRI Classification & SHAP Explanation.

Usage:
  python predict.py --image path/to/mri_scan.jpg --model_path models/best_model.pth
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image

from model import SafeResNet18, build_model, CLASS_NAMES
from utils import get_transforms, denormalize_image
from shap_explain import SafeResNetExplainer, plot_single_explanation


def predict_scan(
    image_path: str,
    model_path: str = "models/best_model.pth",
    output_dir: str = "shap_analysis",
    temperature: float = 5.0,
    device: str = "cpu"
) -> None:
    """Predicts disease stage on a single MRI scan with calibrated probabilities."""
    dev = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # 1. Load Model
    model = build_model(num_classes=4, pretrained=False).to(dev)
    if os.path.exists(model_path):
        model.load_weights(model_path, dev)
        print(f"Loaded trained weights from: {model_path}")
    else:
        print(f"Notice: Checkpoint '{model_path}' not found. Using pretrained backbone.")
        model = build_model(num_classes=4, pretrained=True).to(dev)

    model.eval()

    # 2. Preprocess Input Image
    img = Image.open(image_path).convert("RGB")
    transform = get_transforms(augment=False)
    input_tensor = transform(img).unsqueeze(0).to(dev)

    # 3. Predict Probabilities with Temperature Calibration
    with torch.no_grad():
        logits = model(input_tensor)
        scaled_logits = logits / max(temperature, 1e-4)
        probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

    predicted_label = CLASS_NAMES[pred_class]
    confidence = probs[pred_class]

    def format_conf(prob: float) -> str:
        pct = prob * 100.0
        if pct >= 99.9:
            return "99.9%"
        elif pct <= 0.1 and prob > 0:
            return "<0.1%"
        elif prob == 0:
            return "0.0%"
        return f"{pct:.1f}%"

    print("\n" + "=" * 55)
    print("      ALZHEIMER'S DISEASE MRI DIAGNOSTIC REPORT      ")
    print("=" * 55)
    print(f"Input Scan        : {os.path.abspath(image_path)}")
    print(f"Predicted Stage   : {predicted_label.upper()}")
    print(f"Confidence Score  : {format_conf(confidence)}\n")
    print("Class Probability Breakdown:")
    for c, p in zip(CLASS_NAMES, probs):
        bar = "#" * int(min(p, 0.999) * 30)
        print(f"  - {c:<22}: {format_conf(p):>6} | {bar}")
    print("=" * 55)

    # 4. Generate SHAP & Grad-CAM Explanation
    print("\nComputing SHAP GradientExplainer and Grad-CAM attributions...")
    # Use synthetic background baseline (subtle uniform reference or zero baseline)
    bg_baseline = torch.zeros(10, 3, 224, 224, device=dev)
    explainer = SafeResNetExplainer(model, bg_baseline, dev)

    shap_vals, cam_map, _, _ = explainer.explain_scan(input_tensor)
    display_img = denormalize_image(input_tensor[0])

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_file = os.path.join(output_dir, f"prediction_shap_{base_name}.png")

    plot_single_explanation(
        original_img=display_img,
        shap_vals_all_classes=shap_vals,
        cam_map=cam_map,
        pred_class=pred_class,
        probs=probs,
        output_path=out_file
    )
    print(f"Visual explanation saved to: {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Alzheimer's stage from MRI scan")
    parser.add_argument("--image", type=str, required=True, help="Path to input MRI image")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth", help="Model weights path")
    parser.add_argument("--output_dir", type=str, default="shap_analysis", help="Output directory for explanation")
    parser.add_argument("--temperature", type=float, default=5.0, help="Temperature scaling factor for probability calibration (default: 5.0)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    predict_scan(
        image_path=args.image,
        model_path=args.model_path,
        output_dir=args.output_dir,
        temperature=args.temperature,
        device=args.device
    )
