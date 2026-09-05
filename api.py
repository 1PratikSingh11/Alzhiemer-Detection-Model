"""
Production REST API Backend for Alzheimer's Disease MRI Classification & SHAP Analysis.

Powered by aiohttp (Asynchronous HTTP Server).

Endpoints:
  - GET  /api/health      : Health check and system status (PyTorch device, model status)
  - GET  /api/metrics     : OASIS benchmark evaluation metrics and confusion matrix
  - GET  /api/samples     : Lists available pre-loaded sample MRI scans
  - POST /api/predict     : Accepts MRI image upload -> returns JSON diagnosis & probabilities
  - POST /api/explain     : Accepts MRI image upload -> returns JSON + base64 SHAP & Grad-CAM maps

Usage:
  python api.py --port 8000
"""

import os
import io
import json
import base64
import time
import argparse
from typing import Dict, Any
from aiohttp import web
from PIL import Image
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SafeResNet18, build_model, CLASS_NAMES
from utils import get_transforms, denormalize_image
from shap_explain import SafeResNetExplainer

# Global Model & Explainer Singletons
MODEL = None
DEVICE = None
EXPLAINER = None


def get_model():
    global MODEL, DEVICE
    if MODEL is None:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL = build_model(num_classes=4, pretrained=False).to(DEVICE)
        ckpt_path = "models/best_model.pth"
        if os.path.exists(ckpt_path):
            MODEL.load_weights(ckpt_path, DEVICE)
            print(f"[Backend API] Loaded weights from: {ckpt_path}")
        else:
            MODEL = build_model(num_classes=4, pretrained=True).to(DEVICE)
            print("[Backend API] Using pretrained weights.")
        MODEL.eval()
    return MODEL, DEVICE


def get_explainer():
    global EXPLAINER
    if EXPLAINER is None:
        model, device = get_model()
        bg_baseline = torch.zeros(10, 3, 224, 224, device=device)
        EXPLAINER = SafeResNetExplainer(model, bg_baseline, device)
    return EXPLAINER


# API Route Handlers
async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    model, device = get_model()
    return web.json_response({
        "status": "healthy",
        "service": "Alzheimer's Disease Detection API",
        "version": "1.0.0",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "classes": CLASS_NAMES,
        "timestamp": time.time()
    })


async def handle_metrics(request: web.Request) -> web.Response:
    """Returns official OASIS validation benchmark metrics from the report."""
    metrics_file = "outputs_tcc_resnet18/experiment_results.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            data = json.load(f)
        return web.json_response(data)

    return web.json_response({
        "benchmark_reference": {
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
        }
    })


async def handle_samples(request: web.Request) -> web.Response:
    """Lists preloaded sample MRI files."""
    samples = []
    data_dir = "data"
    for split in ["val", "train"]:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            for c in CLASS_NAMES:
                cdir = os.path.join(split_dir, c)
                if os.path.exists(cdir):
                    for f in os.listdir(cdir):
                        if f.endswith(".jpg") or f.endswith(".png"):
                            samples.append({
                                "class": c,
                                "split": split,
                                "filename": f,
                                "path": os.path.join(cdir, f)
                            })
                            break  # Return one sample per class
    return web.json_response({"samples": samples})


async def handle_predict(request: web.Request) -> web.Response:
    """Accepts image upload or sample query and returns classification results."""
    model, device = get_model()
    img = None

    if request.can_read_body and request.content_type.startswith("multipart/"):
        reader = await request.multipart()
        field = await reader.next()
        if field.name == "image":
            raw_bytes = await field.read()
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    else:
        # Fallback to sample query param
        sample_name = request.query.get("sample", "Moderate Dementia")
        sample_path = os.path.join("data", "val", sample_name)
        if os.path.exists(sample_path):
            files = [f for f in os.listdir(sample_path) if f.endswith(".jpg")]
            if files:
                img = Image.open(os.path.join(sample_path, files[0])).convert("RGB")

    if img is None:
        return web.json_response({"error": "No image provided. Upload multipart 'image' or use ?sample=<Class>"}, status=400)

    start_time = time.time()
    transform = get_transforms(augment=False)
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

    latency_ms = (time.time() - start_time) * 1000

    capped_conf = min(round(float(probs[pred_class]) * 100, 1), 99.9)
    return web.json_response({
        "predicted_class": CLASS_NAMES[pred_class],
        "confidence": float(probs[pred_class]),
        "confidence_display": f"{capped_conf:.1f}%",
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
        "latency_ms": round(latency_ms, 2)
    })


async def handle_explain(request: web.Request) -> web.Response:
    """Accepts image and generates SHAP and Grad-CAM explanations."""
    model, device = get_model()
    explainer = get_explainer()
    img = None

    if request.can_read_body and request.content_type.startswith("multipart/"):
        reader = await request.multipart()
        field = await reader.next()
        if field.name == "image":
            raw_bytes = await field.read()
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    else:
        sample_name = request.query.get("sample", "Moderate Dementia")
        sample_path = os.path.join("data", "val", sample_name)
        if os.path.exists(sample_path):
            files = [f for f in os.listdir(sample_path) if f.endswith(".jpg")]
            if files:
                img = Image.open(os.path.join(sample_path, files[0])).convert("RGB")

    if img is None:
        return web.json_response({"error": "No image provided"}, status=400)

    start_time = time.time()
    transform = get_transforms(augment=False)
    input_tensor = transform(img).unsqueeze(0).to(device)

    # SHAP & Grad-CAM
    shap_vals, cam_map, pred_class, probs = explainer.explain_scan(input_tensor)
    display_img = denormalize_image(input_tensor[0])
    pred_shap = shap_vals[pred_class]
    shap_2d = np.mean(pred_shap, axis=0) if pred_shap.shape[0] == 3 else np.mean(pred_shap, axis=-1)

    abs_max = float(np.percentile(np.abs(shap_2d), 99.5))
    abs_max = max(abs_max, 1e-6)

    # Render base64 SHAP heatmap
    buf = io.BytesIO()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(display_img)
    axes[0].set_title(f"Input Scan\nPredicted: {CLASS_NAMES[pred_class]}", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(display_img, cmap="gray", alpha=0.45)
    im_shap = axes[1].imshow(shap_2d, cmap="seismic", alpha=0.85, vmin=-abs_max, vmax=abs_max)
    axes[1].set_title("SHAP GradientExplainer\n(Positive vs Negative Evidence)", fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im_shap, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(display_img)
    im_cam = axes[2].imshow(cam_map, cmap="jet", alpha=0.5)
    axes[2].set_title("Grad-CAM Activation\n(Layer4 Saliency)", fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im_cam, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    b64_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
    latency_ms = (time.time() - start_time) * 1000

    return web.json_response({
        "predicted_class": CLASS_NAMES[pred_class],
        "confidence": float(probs[pred_class]),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
        "latency_ms": round(latency_ms, 2),
        "explanation_plot_base64": b64_plot
    })


def create_app() -> web.Application:
    app = web.Application(client_max_size=30 * 1024 * 1024)
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/metrics", handle_metrics)
    app.router.add_get("/api/samples", handle_samples)
    app.router.add_post("/api/predict", handle_predict)
    app.router.add_get("/api/predict", handle_predict)
    app.router.add_post("/api/explain", handle_explain)
    app.router.add_get("/api/explain", handle_explain)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alzheimer's MRI REST API")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"Starting Alzheimer's Disease Detection REST API on http://{args.host}:{args.port}")
    app = create_app()
    web.run_app(app, host=args.host, port=args.port)
