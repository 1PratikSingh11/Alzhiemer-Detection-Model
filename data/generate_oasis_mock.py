"""
Synthetic OASIS MRI Data Generator.

Generates structurally realistic 2D axial T1-weighted brain MRI slice simulations
across the 4 Alzheimer's Disease stages:
  1. Non-Demented: Normal brain volume, small lateral ventricles, dense parenchyma.
  2. Very Mild Dementia: Early ventricular enlargement, mild sulcal prominence.
  3. Mild Dementia: Clear bilateral ventricular dilation, cortical volume loss.
  4. Moderate Dementia: Severe ex-vacuo ventricular dilation, marked cortical atrophy.

This allows immediate out-of-the-box pipeline verification, training execution,
and SHAP GradientExplainer testing before connecting the complete 86k OASIS dataset.
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_synthetic_mri_slice(
    stage: str,
    width: int = 496,
    height: int = 248,
    seed: int = None
) -> Image.Image:
    """
    Renders an anatomically inspired T1-weighted axial MRI brain slice.
    """
    if seed is not None:
        np.random.seed(seed)

    # Base image: Black background (Air / background scanner noise)
    canvas = np.random.normal(loc=8, scale=3, size=(height, width)).astype(np.float32)
    canvas = np.clip(canvas, 0, 255)

    img = Image.fromarray(canvas.astype(np.uint8)).convert("L")
    draw = ImageDraw.Draw(img)

    center_x = width // 2
    center_y = height // 2
    
    # Random slight variations in head dimensions
    rx = int(width * 0.38 + np.random.uniform(-5, 5))
    ry = int(height * 0.42 + np.random.uniform(-5, 5))

    # 1. Skull and Scalp (outer hyperintense / hypointense ring)
    skull_box = [center_x - rx, center_y - ry, center_x + rx, center_y + ry]
    draw.ellipse(skull_box, fill=45, outline=120, width=4)

    # 2. Subdural / CSF boundary (dark ring)
    dura_rx, dura_ry = rx - 6, ry - 6
    dura_box = [center_x - dura_rx, center_y - dura_ry, center_x + dura_rx, center_y + dura_ry]
    draw.ellipse(dura_box, fill=25, outline=20, width=2)

    # Stage-dependent atrophy factor
    # Higher stage = greater atrophy (thinner cortex, bigger ventricles)
    atrophy_factors = {
        "Non-Demented": 0.0,
        "Very Mild Dementia": 0.25,
        "Mild Dementia": 0.60,
        "Moderate Dementia": 0.95
    }
    atrophy = atrophy_factors.get(stage, 0.0)

    # 3. Brain Parenchyma (Cortical Gray Matter + White Matter)
    brain_rx = int(dura_rx - 4 - (atrophy * 10))
    brain_ry = int(dura_ry - 4 - (atrophy * 10))
    brain_box = [center_x - brain_rx, center_y - brain_ry, center_x + brain_rx, center_y + brain_ry]
    
    # Gray Matter (intensity ~110-130)
    draw.ellipse(brain_box, fill=115, outline=100, width=3)

    # White Matter (higher T1 signal intensity ~170-190)
    wm_rx = int(brain_rx * 0.82)
    wm_ry = int(brain_ry * 0.82)
    wm_box = [center_x - wm_rx, center_y - wm_ry, center_x + wm_rx, center_y + wm_ry]
    draw.ellipse(wm_box, fill=175, outline=150, width=4)

    # 4. Deep Gray Nuclei (Thalamus / Basal Ganglia)
    bg_rx, bg_ry = int(wm_rx * 0.45), int(wm_ry * 0.45)
    bg_box = [center_x - bg_rx, center_y - bg_ry, center_x + bg_rx, center_y + bg_ry]
    draw.ellipse(bg_box, fill=135)

    # 5. Ventricular System (Lateral Ventricles: Hypointense CSF, ~15-30 intensity)
    # Lateral ventricles dramatically enlarge with Alzheimer's disease (hydrocephalus ex vacuo)
    base_vent_w = int(12 + (atrophy * 28) + np.random.uniform(-1, 2))
    base_vent_h = int(28 + (atrophy * 38) + np.random.uniform(-2, 2))
    vent_sep = int(10 + (atrophy * 8))

    # Left lateral ventricle anterior/posterior horn
    left_vent_box = [
        center_x - vent_sep - base_vent_w,
        center_y - (base_vent_h // 2),
        center_x - vent_sep,
        center_y + (base_vent_h // 2)
    ]
    draw.ellipse(left_vent_box, fill=22, outline=15, width=1)

    # Right lateral ventricle
    right_vent_box = [
        center_x + vent_sep,
        center_y - (base_vent_h // 2),
        center_x + vent_sep + base_vent_w,
        center_y + (base_vent_h // 2)
    ]
    draw.ellipse(right_vent_box, fill=22, outline=15, width=1)

    # Third ventricle (midline slit, widens with dementia)
    third_w = int(3 + (atrophy * 7))
    third_h = int(16 + (atrophy * 12))
    third_box = [
        center_x - (third_w // 2),
        center_y - (third_h // 2),
        center_x + (third_w // 2),
        center_y + (third_h // 2)
    ]
    draw.rectangle(third_box, fill=20)

    # 6. Apply realistic Gaussian blur to simulate MRI point-spread function
    blurred = img.filter(ImageFilter.GaussianBlur(radius=1.8))
    arr = np.array(blurred, dtype=np.float32)

    # 7. Add smooth MRI B1 bias field (intensity inhomogeneity artifact)
    yy, xx = np.mgrid[:height, :width]
    bias = 1.0 + 0.08 * np.sin(xx / 80.0) * np.cos(yy / 60.0)
    arr = arr * bias

    # 8. Add Rician / Gaussian scanner noise
    noise = np.random.normal(loc=0, scale=3.5, size=(height, width))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def generate_dataset(
    output_root: str = "data",
    samples_per_class_train: int = 40,
    samples_per_class_val: int = 10
):
    """Generates balanced/representative sample dataset split into train/ and val/."""
    classes = [
        "Non-Demented",
        "Very Mild Dementia",
        "Mild Dementia",
        "Moderate Dementia"
    ]

    print(f"Generating synthetic OASIS MRI dataset in: {output_root}")
    for split, count in [("train", samples_per_class_train), ("val", samples_per_class_val)]:
        for c in classes:
            class_dir = os.path.join(output_root, split, c)
            os.makedirs(class_dir, exist_ok=True)
            print(f"  -> Generating {count} scans for [{split}/{c}]...")
            for i in range(count):
                img = create_synthetic_mri_slice(stage=c, width=496, height=248, seed=i + 1000)
                img.save(os.path.join(class_dir, f"oasis_{split}_{c.replace(' ', '_')}_{i:04d}.jpg"))

    print("[Done] Synthetic OASIS dataset generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OASIS MRI dataset")
    parser.add_argument("--output_root", type=str, default="data", help="Output directory")
    parser.add_argument("--train_samples", type=int, default=40, help="Train samples per class")
    parser.add_argument("--val_samples", type=int, default=10, help="Val samples per class")
    args = parser.parse_args()

    generate_dataset(
        output_root=args.output_root,
        samples_per_class_train=args.train_samples,
        samples_per_class_val=args.val_samples
    )
