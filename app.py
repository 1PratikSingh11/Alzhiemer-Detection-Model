# app.py
"""
Premium Streamlit GUI for NeuroScan AI (Alzheimer MRI) — SHAP only (no Captum)
Place at project root. Assumes:
 - models/resnet18_model.pth
 - utils/gradcam.py
 - utils/difficulty.py
 - utils/mri_explainer.py
 - utils/shap_manager.py
 - model_definition.py (SafeResNet18)
"""

import os
# pragmatic workaround for OpenMP duplicate runtime on Windows (see console warnings)
# NOTE: This is a workaround — if you can remove duplicate OpenMP installs, do that for production.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import time
from pathlib import Path

# -------------------------
# App configuration & paths
# -------------------------
st.set_page_config(page_title="Disease Detection Model", layout="wide", page_icon="🧠")

PROJECT_ROOT = Path(".")
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_model.pth"
SAMPLE_DIR = PROJECT_ROOT / "sample_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SHAP_OUTPUT_DIR = OUTPUT_DIR / "shap_analysis"
GRADCAM_OUTPUT_DIR = OUTPUT_DIR / "gradcam_analysis"
MRI_OUTPUT_DIR = OUTPUT_DIR / "mri_explanations"

for p in (OUTPUT_DIR, SHAP_OUTPUT_DIR, GRADCAM_OUTPUT_DIR, MRI_OUTPUT_DIR):
    os.makedirs(p, exist_ok=True)

CLASS_NAMES = ['Non Demented', 'Very mild Dementia', 'Mild Dementia', 'Moderate Dementia']

# -------------------------
# Premium CSS
# -------------------------
def load_css():
    """
    Production-grade CSS styling for Alzheimer's Detection Platform.
    Implements medical-grade UI patterns with accessibility considerations.
    """
    st.markdown(
        """
        <style>
        /* ═══════════════════════════════════════════════════════════════
           TYPOGRAPHY & FOUNDATIONAL VARIABLES
           ═══════════════════════════════════════════════════════════════ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            /* Medical-grade color palette - trust & professionalism */
            --bg-primary: #0a0e17;
            --bg-secondary: #0f1419;
            --bg-elevated: #151a23;
            
            /* Glass morphism layers */
            --glass-subtle: rgba(255, 255, 255, 0.02);
            --glass-light: rgba(255, 255, 255, 0.04);
            --glass-medium: rgba(255, 255, 255, 0.06);
            --glass-strong: rgba(255, 255, 255, 0.08);
            
            /* Semantic colors */
            --text-primary: #e5e9f0;
            --text-secondary: #8a9bb0;
            --text-tertiary: #606c7e;
            
            /* Medical-appropriate accent colors */
            --accent-primary: #5e81ac;      /* Trust blue */
            --accent-secondary: #88c0d0;    /* Calm cyan */
            --accent-success: #a3be8c;      /* Health green */
            --accent-warning: #ebcb8b;      /* Alert amber */
            --accent-critical: #bf616a;     /* Critical red */
            
            /* Elevation & depth */
            --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.2);
            --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.3);
            --shadow-lg: 0 16px 48px rgba(0, 0, 0, 0.4);
            --shadow-xl: 0 24px 64px rgba(0, 0, 0, 0.5);
            
            /* Animation durations */
            --duration-fast: 0.15s;
            --duration-base: 0.25s;
            --duration-slow: 0.4s;
            
            /* Border radius system */
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
            --radius-xl: 18px;
        }

        /* ═══════════════════════════════════════════════════════════════
           ANIMATION LIBRARY
           ═══════════════════════════════════════════════════════════════ */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(-15px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }

        /* ═══════════════════════════════════════════════════════════════
           BASE APPLICATION STRUCTURE
           ═══════════════════════════════════════════════════════════════ */
        .stApp {
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(94, 129, 172, 0.06) 0%, transparent 45%),
                radial-gradient(circle at 90% 80%, rgba(136, 192, 208, 0.04) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(163, 190, 140, 0.02) 0%, transparent 50%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: var(--text-primary);
            font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11';
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        .block-container {
            padding: 3rem 2rem;
            max-width: 1200px;
            animation: fadeIn var(--duration-slow) ease-out;
        }

        /* ═══════════════════════════════════════════════════════════════
           GLASS CARD SYSTEM - Medical Context
           ═══════════════════════════════════════════════════════════════ */
        .glass-card {
            position: relative;
            background: linear-gradient(
                135deg,
                var(--glass-subtle) 0%,
                var(--glass-light) 100%
            );
            border: 1px solid var(--glass-light);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            padding: 28px;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-md);
            transition: all var(--duration-base) cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeInUp 0.6s ease-out forwards;
            overflow: hidden;
        }
        
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(
                90deg,
                transparent,
                var(--accent-secondary),
                transparent
            );
            opacity: 0;
            transition: opacity var(--duration-base);
        }
        
        .glass-card:hover {
            border-color: var(--glass-medium);
            background: linear-gradient(
                135deg,
                var(--glass-light) 0%,
                var(--glass-medium) 100%
            );
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }
        
        .glass-card:hover::before {
            opacity: 0.5;
        }
        
        /* Card variants for different contexts */
        .glass-card-info {
            border-left: 3px solid var(--accent-primary);
        }
        
        .glass-card-success {
            border-left: 3px solid var(--accent-success);
        }
        
        .glass-card-warning {
            border-left: 3px solid var(--accent-warning);
        }
        
        .glass-card-critical {
            border-left: 3px solid var(--accent-critical);
        }

        /* ═══════════════════════════════════════════════════════════════
           TYPOGRAPHY SYSTEM
           ═══════════════════════════════════════════════════════════════ */
        .title {
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #ffffff;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #ffffff 0%, var(--text-primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: fadeInUp 0.5s ease-out;
        }
        
        .subtitle {
            color: var(--text-secondary);
            font-size: 1.05rem;
            font-weight: 400;
            line-height: 1.6;
            max-width: 600px;
            animation: fadeInUp 0.6s ease-out 0.1s backwards;
        }
        
        .section-header {
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 16px;
            letter-spacing: -0.02em;
        }
        
        .metric-label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 4px;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
        }

        /* ═══════════════════════════════════════════════════════════════
           FORM CONTROLS & INPUTS
           ═══════════════════════════════════════════════════════════════ */
        .stButton > button {
            position: relative;
            border-radius: var(--radius-md);
            padding: 0.65rem 1.5rem;
            background: var(--glass-light) !important;
            border: 1px solid var(--glass-medium) !important;
            color: var(--text-primary) !important;
            font-weight: 500;
            font-size: 0.95rem;
            transition: all var(--duration-base) cubic-bezier(0.4, 0, 0.2, 1);
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: var(--accent-primary);
            opacity: 0.1;
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .stButton > button:hover {
            background: var(--glass-medium) !important;
            border-color: var(--accent-primary) !important;
            transform: translateY(-2px);
            box-shadow: var(--shadow-sm);
        }
        
        .stButton > button:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Primary action button */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
            border: none !important;
            color: #ffffff !important;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(94, 129, 172, 0.3);
        }
        
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(94, 129, 172, 0.4);
            transform: translateY(-2px);
        }
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            background: var(--glass-subtle) !important;
            border: 1px solid var(--glass-light) !important;
            border-radius: var(--radius-md) !important;
            color: var(--text-primary) !important;
            padding: 0.6rem 0.9rem !important;
            transition: all var(--duration-base);
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            border-color: var(--accent-primary) !important;
            background: var(--glass-light) !important;
            box-shadow: 0 0 0 3px rgba(94, 129, 172, 0.1) !important;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background: var(--glass-subtle);
            border: 2px dashed var(--glass-medium);
            border-radius: var(--radius-lg);
            padding: 2rem;
            transition: all var(--duration-base);
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: var(--accent-primary);
            background: var(--glass-light);
        }
        /* ═══════════════════════════════════════════════════════════════
           TAB NAVIGATION SPANNING
           ═══════════════════════════════════════════════════════════════ */
        .stTabs [role="tablist"] {
            display: flex;
            width: 100%;
            gap: 2px;
            
        }

        .stTabs [role="tab"] {
            flex: 1;
            text-align: center;
            justify-content: center;
            padding: 12px 0px;
            transition: all var(--duration-base) ease;
            border-radius: var(--radius-md);
        }

        .stTabs [role="tab"]:hover {
            background: var(--glass-subtle);
            color: var(--accent-primary) !important;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            color: var(--accent-primary) !important;
            border-bottom-color: var(--accent-primary) !important;
            background: rgba(94, 129, 172, 0.03);
        }
        /* ═══════════════════════════════════════════════════════════════
           SIDEBAR NAVIGATION
           ═══════════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
            border-right: 1px solid var(--glass-light);
            box-shadow: var(--shadow-md);
        }
        
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }
        
        [data-testid="stSidebar"] .element-container {
            animation: slideInRight 0.4s ease-out backwards;
        }
        
        [data-testid="stSidebar"] .element-container:nth-child(2) {
            animation-delay: 0.1s;
        }
        
        [data-testid="stSidebar"] .element-container:nth-child(3) {
            animation-delay: 0.2s;
        }

        /* ═══════════════════════════════════════════════════════════════
           DATA VISUALIZATION ENHANCEMENTS
           ═══════════════════════════════════════════════════════════════ */
        .stPlotlyChart {
            background: var(--glass-subtle);
            border-radius: var(--radius-lg);
            padding: 1rem;
            border: 1px solid var(--glass-light);
        }
        
        /* Dataframe styling */
        [data-testid="stDataFrame"] {
            border-radius: var(--radius-md);
            overflow: hidden;
            border: 1px solid var(--glass-light);
        }

        /* ═══════════════════════════════════════════════════════════════
           ALERT & NOTIFICATION COMPONENTS
           ═══════════════════════════════════════════════════════════════ */
        .stAlert {
            background: var(--glass-light) !important;
            border-radius: var(--radius-md) !important;
            border-left: 4px solid var(--accent-primary) !important;
            backdrop-filter: blur(10px);
            animation: fadeInUp 0.4s ease-out;
        }
        
        .stSuccess {
            border-left-color: var(--accent-success) !important;
        }
        
        .stWarning {
            border-left-color: var(--accent-warning) !important;
        }
        
        .stError {
            border-left-color: var(--accent-critical) !important;
        }

        /* ═══════════════════════════════════════════════════════════════
           PROGRESS & LOADING STATES
           ═══════════════════════════════════════════════════════════════ */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)) !important;
            border-radius: var(--radius-sm);
        }
        
        .stSpinner > div {
            border-top-color: var(--accent-primary) !important;
        }

        /* ═══════════════════════════════════════════════════════════════
           SCROLLBAR CUSTOMIZATION
           ═══════════════════════════════════════════════════════════════ */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--glass-medium);
            border-radius: var(--radius-sm);
            border: 2px solid var(--bg-primary);
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--glass-strong);
        }

        /* ═══════════════════════════════════════════════════════════════
           ACCESSIBILITY & FOCUS STATES
           ═══════════════════════════════════════════════════════════════ */
        *:focus-visible {
            outline: 2px solid var(--accent-primary);
            outline-offset: 2px;
            border-radius: var(--radius-sm);
        }

        /* ═══════════════════════════════════════════════════════════════
           RESPONSIVE DESIGN
           ═══════════════════════════════════════════════════════════════ */
        @media (max-width: 768px) {
            .block-container {
                padding: 1.5rem 1rem;
            }
            
            .title {
                font-size: 2rem;
            }
            
            .glass-card {
                padding: 20px;
            }
        }

        /* ═══════════════════════════════════════════════════════════════
           UTILITY CLASSES
           ═══════════════════════════════════════════════════════════════ */
        .fade-in {
            animation: fadeIn var(--duration-base) ease-out;
        }
        
        .slide-in-right {
            animation: slideInRight var(--duration-base) ease-out;
        }
        
        .text-gradient {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .monospace {
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-variant-ligatures: none;
        }
        
        /* Divider */
        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--glass-medium), transparent);
            margin: 2rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
load_css()

# -------------------------
# Transforms (same as training)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((248, 496)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Load model and utils (cached)
# -------------------------
@st.cache_resource
def load_model_and_utils():
    info = {"model": None, "gradcam": None, "difficulty": None, "mri_explainer": None, "shap_manager": None, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    device = info["device"]
    # load model_definition
    try:
        from model_definition import SafeResNet18
        model = SafeResNet18(num_classes=4).to(device)
        if MODEL_PATH.exists():
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
        else:
            # still provide model object (random weights) but warn in UI
            model.eval()
        info["model"] = model
    except Exception as e:
        st.error(f"Error loading model_definition or weights: {e}")
        return info

    # load utilities if available (fail gracefully)
    try:
        from utils.gradcam import GradCAMAnalyzer
        info["gradcam"] = GradCAMAnalyzer(model, device, CLASS_NAMES)
    except Exception as e:
        st.warning(f"GradCAM not available: {e}")

    try:
        from utils.difficulty import DifficultyAnalyzer
        info["difficulty"] = DifficultyAnalyzer(model, device, CLASS_NAMES)
    except Exception as e:
        st.warning(f"Difficulty analyzer not available: {e}")

    try:
        from utils.mri_explainer import MRIAlzheimerExplainer
        info["mri_explainer"] = MRIAlzheimerExplainer(model, device, CLASS_NAMES)
    except Exception as e:
        st.warning(f"MRI explainer not available: {e}")

    try:
        from utils.shap_manager import SHAPInteractiveManager
        info["shap_manager"] = SHAPInteractiveManager(model, device, CLASS_NAMES, str(SHAP_OUTPUT_DIR))
    except Exception as e:
        # shap_manager should handle shap import internally and fallback if shap missing
        st.warning(f"SHAP manager unavailable: {e}")

    return info

helpers = load_model_and_utils()
model = helpers.get("model")
gradcam = helpers.get("gradcam")
difficulty = helpers.get("difficulty")
mri_explainer = helpers.get("mri_explainer")
shap_manager = helpers.get("shap_manager")
DEVICE = helpers.get("device")

# -------------------------
# Top header
# -------------------------
st.markdown("<div style='display:flex;align-items:center;gap:18px'>", unsafe_allow_html=True)
st.markdown("<div><h1 class='title'>Alzheimer’s Disease Detection Model</h1><div class='subtitle'>Alzheimer MRI analysis — ResNet18 · Explainable AI (Grad-CAM & SHAP)</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Configuration")
st.sidebar.caption("Toggle explainability and input options")

show_gradcam = st.sidebar.checkbox("Grad-CAM heatmap", True)
show_shap = st.sidebar.checkbox("SHAP explainability", False)
show_mri = st.sidebar.checkbox("Clinical explanation", True)
show_difficulty = st.sidebar.checkbox("Difficulty hint", True)

uploaded = st.sidebar.file_uploader("Upload MRI scan (jpg/png)", type=["jpg", "png", "jpeg"])
sample_files = sorted([p for p in (SAMPLE_DIR.exists() and SAMPLE_DIR or []).glob("*.*")] ) if SAMPLE_DIR.exists() else []
if sample_files:
    sel = st.sidebar.selectbox("Or choose sample", ["-- none --"] + [p.name for p in sample_files])
    if sel != "-- none --" and uploaded is None:
        uploaded = open(SAMPLE_DIR / sel, "rb")

# helper to preprocess
def load_and_preprocess(fileobj):
    img = Image.open(fileobj).convert("RGB")
    tensor = transform(img)
    return img, tensor

# Main
if not uploaded:
    st.markdown("<div class='glass-card' style='text-align:center'><h3>Ready to analyze</h3><p class='small-muted'>Upload an MRI on the left to start — the app will show prediction, Grad-CAM and SHAP results.</p></div>", unsafe_allow_html=True)
else:
    try:
        img, tensor = load_and_preprocess(uploaded)
        filename = getattr(uploaded, "name", f"uploaded_{int(time.time())}.png")

        # Top row: image + summary
        col1, col2 = st.columns([1.2, 2])
        with col1:
            st.image(img, caption="Input MRI", use_container_width=True)

        # Prediction
        if model is None:
            st.error("Model not loaded — place model weights at 'models/resnet18_model.pth'.")
            pred_idx = None
            conf_val = None
        else:
            model.eval()
            with torch.no_grad():
                batch = tensor.unsqueeze(0).to(DEVICE)
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                pred_idx = int(pred.item())
                conf_val = float(conf.item())

        with col2:
            if pred_idx is not None:
                badge_color = "#4ade80" if pred_idx == 0 else ("#facc15" if pred_idx == 1 else "#fb7185")
                st.markdown(f"<div class='glass-card'><h3>Result</h3><div><span class='badge' style='padding: 4px 8px;border-radius: 5px; background:{badge_color};color:black'>{CLASS_NAMES[pred_idx]}</span> <span class='small-muted' style='margin-left:12px'>Confidence: {conf_val:.2%}</span></div><p class='small-muted' style='margin-top:10px'>Model: ResNet18</p></div>", unsafe_allow_html=True)
            else:
                st.info("No prediction available")

        # Tabs: Grad-CAM, SHAP, MRI report, Difficulty
        tab1, tab2, tab3, tab4 = st.tabs(["Grad-CAM", "SHAP", "Clinical report", "Difficulty"])

        # Grad-CAM
        with tab1:
            st.subheader("Grad-CAM")
            if gradcam is None:
                st.warning("Grad-CAM utility not loaded.")
            elif not show_gradcam:
                st.info("Enable Grad-CAM in the sidebar.")
            else:
                try:
                    heatmap, _ = gradcam.generate_heatmap(tensor, pred_idx)
                    fig = gradcam.visualize_gradcam(tensor, filename, target_class=pred_idx, save_path=str(GRADCAM_OUTPUT_DIR / f"gradcam_{Path(filename).stem}.png"))
                    st.pyplot(fig)
                    # download
                    saved = GRADCAM_OUTPUT_DIR / f"gradcam_{Path(filename).stem}.png"
                    if saved.exists():
                        with open(saved, "rb") as fh:
                            st.download_button("Download Grad-CAM PNG", fh.read(), file_name=saved.name, mime="image/png")
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        # SHAP
        with tab2:
            st.subheader("SHAP Explainability")
            if shap_manager is None:
                st.warning("SHAP manager not loaded.")
            elif not show_shap:
                st.info("Enable SHAP in the sidebar (may be slow).")
            else:
                with st.spinner("Running SHAP (or fallback) — this may take a few seconds..."):
                    try:
                        # Prepare background/test if needed
                        if hasattr(shap_manager, "prepare_shap_data_from_single_image"):
                            shap_manager.prepare_shap_data_from_single_image(tensor.unsqueeze(0))
                        result = shap_manager.analyze_single_image(tensor, filename, save_prefix=f"shap_{Path(filename).stem}")
                        # show image
                        if "shap_file" in result and Path(result["shap_file"]).exists():
                            st.image(result["shap_file"], caption="SHAP result", use_container_width=True)
                            with open(result["shap_file"], "rb") as fh:
                                st.download_button("Download SHAP PNG", fh.read(), file_name=Path(result["shap_file"]).name, mime="image/png")
                        else:
                            st.info("SHAP produced no PNG output — check logs.")
                    except Exception as e:
                        st.error(f"SHAP analysis failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Clinical report
        with tab3:
            st.subheader("Clinical-style Explanation")
            if mri_explainer is None:
                st.warning("MRI explainer not available.")
            elif not show_mri:
                st.info("Enable clinical explanation in the sidebar.")
            else:
                try:
                    # explain and save figure into MRI_OUTPUT_DIR
                    mri_explainer.explain_mri_findings(tensor, filename, pred_idx)
                    expl = MRI_OUTPUT_DIR / f"mri_explanation_{CLASS_NAMES[pred_idx].replace(' ', '_')}.png"
                    if expl.exists():
                        st.image(str(expl), caption="Clinical Explanation", use_container_width=True)
                        with open(expl, "rb") as fh:
                            st.download_button("Download Clinical PNG", fh.read(), file_name=expl.name, mime="image/png")
                    else:
                        st.info("No clinical figure generated.")
                except Exception as e:
                    st.error(f"MRI explainer error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Difficulty
        with tab4:
            st.subheader("Model Certainty & Difficulty")
            if difficulty is None:
                st.warning("Difficulty analyzer not loaded.")
            else:
                try:
                    if conf_val is not None:
                        if conf_val > 0.8:
                            st.success("Easy: High confidence")
                        elif conf_val < 0.6:
                            st.error("Hard: Low confidence — recommend expert review")
                        else:
                            st.warning("Medium difficulty")
                        st.metric("Confidence", f"{conf_val:.4f}")
                except Exception as e:
                    st.error(f"Difficulty failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Footer JSON
        with st.expander("Download JSON report"):
            report = {"image": filename, "prediction": CLASS_NAMES[pred_idx] if pred_idx is not None else None,
                      "confidence": conf_val, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            st.json(report)
            st.download_button("Download report JSON", data=json.dumps(report, indent=2), file_name=f"report_{Path(filename).stem}.json")

    except Exception as e:
        st.error(f"Unhandled error: {e}")
