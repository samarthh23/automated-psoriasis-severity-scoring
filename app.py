"""
Psoriasis Lesion Segmentation — Clinical Analysis Interface
Refined dark theme with bug-fixed metrics and clean layout.
"""

import os
import streamlit as st
import torch
import cv2
import numpy as np
import time
from datetime import datetime

import config
from segmentation_model import UNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Psoriasis Lesion Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    background-color: #0b0f1a;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1280px; }

/* ── Wordmark / brand header ── */
.brand-wrap {
    display: flex;
    align-items: flex-end;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
}
.brand-title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #f1f5f9;
    line-height: 1;
}
.brand-badge {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #818cf8;
    background: rgba(129,140,248,0.12);
    border: 1px solid rgba(129,140,248,0.25);
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    margin-bottom: 0.2rem;
}
.brand-subtitle {
    font-size: 0.875rem;
    color: #64748b;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
}

/* ── Metrics row ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #1e293b;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.5rem;
    border: 1px solid #1e293b;
}
.metric-cell {
    background: #0f172a;
    padding: 1.25rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748b;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}
.metric-value.accent { color: #818cf8; }
.metric-value.good   { color: #34d399; }
.metric-value.warn   { color: #fbbf24; }
.metric-value.danger { color: #f87171; }
.metric-sub {
    font-size: 0.75rem;
    color: #475569;
}

/* ── Override Streamlit's metric widget ── */
div[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
    color: #f1f5f9 !important;
    background: none !important;
    -webkit-text-fill-color: #f1f5f9 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
div[data-testid="metric-container"] {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
}

/* ── Result badge / severity pill ── */
.sev-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
}
.sev-low    { background: rgba(52,211,153,0.12); color:#34d399; border:1px solid rgba(52,211,153,0.25); }
.sev-medium { background: rgba(251,191,36,0.12);  color:#fbbf24; border:1px solid rgba(251,191,36,0.25); }
.sev-high   { background: rgba(248,113,113,0.12); color:#f87171; border:1px solid rgba(248,113,113,0.25); }

/* ── Status / result bar at top of results ── */
.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.result-header-left {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}
.result-header-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748b;
}
.result-header-file {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
}
.result-header-ts {
    font-size: 0.75rem;
    color: #475569;
}

/* ── Image panel label ── */
.img-label {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #475569;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: #4f46e5;
    color: white;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.875rem;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1rem;
    transition: background 0.2s ease, transform 0.15s ease;
    letter-spacing: 0.2px;
}
.stButton > button:hover {
    background: #6366f1;
    transform: translateY(-1px);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e293b !important;
    gap: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px;
    padding: 0.5rem 1rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom: 2px solid #818cf8 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e293b !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    font-size: 0.825rem;
}

/* ── Sidebar model info box ── */
.model-info {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    font-size: 0.8rem;
    line-height: 1.8;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}
.model-info span { color: #818cf8; }

/* ── Upload zone ── */
div[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #1e293b !important;
    background: #0f172a !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease, background 0.2s ease;
}
div[data-testid="stFileUploadDropzone"]:hover {
    border-color: #4f46e5 !important;
    background: rgba(79,70,229,0.04) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* ── Welcome state ── */
.welcome-panel {
    padding: 3rem 2.5rem;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    text-align: center;
    margin-top: 1rem;
}
.welcome-panel h2 {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
}
.welcome-panel p {
    font-size: 0.875rem;
    color: #64748b;
    max-width: 420px;
    margin: 0 auto 1.75rem;
    line-height: 1.6;
}
.steps-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
}
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
}
.step-num {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: rgba(79,70,229,0.15);
    border: 1px solid rgba(79,70,229,0.35);
    color: #818cf8;
    font-size: 0.875rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}
.step-text {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
}

/* ── Info callout ── */
.callout {
    background: rgba(79,70,229,0.06);
    border: 1px solid rgba(79,70,229,0.2);
    border-left: 3px solid #4f46e5;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.1rem;
    font-size: 0.825rem;
    color: #94a3b8;
    line-height: 1.6;
    margin-top: 1rem;
}

/* ── Debug info ── */
.debug-row {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #64748b;
    line-height: 1.9;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────────────────────
IMG_SIZE = config.IMG_SIZE
DEVICE = config.DEVICE
MODEL_PATH = config.get_model_path()
MODEL_FILENAME = os.path.basename(MODEL_PATH)   # BUG FIX: was MODEL_PATH.split("\\\\") — never matched on Windows


# ── Model loading ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = UNet().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        # weights_only=False required for PyTorch 2.6+ with custom pth files
    )
    model.eval()
    return model


# ── Helpers ─────────────────────────────────────────────────────────────────────
def get_severity_level(severity: float):
    if severity < 10:
        return "Low",    "sev-low",    "✓"
    elif severity < 30:
        return "Moderate", "sev-medium", "⚠"
    else:
        return "High",   "sev-high",   "✕"


def process_image(uploaded_file, model, threshold: float = 0.5):
    """Run segmentation and return all metrics + visualizations."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Preprocessing — matches training: divide by 255, no ImageNet normalisation
    input_img = img_resized / 255.0
    tensor = torch.tensor(
        np.transpose(input_img, (2, 0, 1)), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    # Inference
    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)[0, 0].cpu().numpy()
        prob = torch.sigmoid(torch.from_numpy(logits)).numpy()  # [0, 1] probabilities
    inference_time = time.time() - t0

    # Binary mask
    mask = (prob > threshold).astype(np.uint8)

    # Severity — percentage of pixels classified as lesion
    lesion_pixels = int(np.sum(mask))
    total_pixels  = mask.size
    severity = round((lesion_pixels / total_pixels) * 100, 2)

    # --- BUG FIX: Dice used to compare mask to itself → always 100% ---
    # Now we show "Model Confidence": mean sigmoid probability for lesion pixels.
    # This is a meaningful signal: how certain the model is about its detections.
    if lesion_pixels > 0:
        confidence = float(np.mean(prob[mask == 1])) * 100
    else:
        confidence = 0.0
    confidence = round(confidence, 1)

    # Visualisations
    mask_display = (mask * 255).astype(np.uint8)

    overlay = img_resized.copy()
    overlay[mask == 1] = [220, 76, 100]  # subdued rose rather than pure red

    blended = cv2.addWeighted(img_resized, 0.65, overlay, 0.35, 0)

    # Grad-CAM
    target_layers = [model.conv1.conv[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(
        input_tensor=tensor,
        targets=[lambda o: o.mean()]
    )[0]
    heatmap = show_cam_on_image(input_img, grayscale_cam, use_rgb=True)

    return {
        "original":       img_resized,
        "mask":           mask_display,
        "overlay":        overlay,
        "blended":        blended,
        "heatmap":        heatmap,
        "prob_map":       prob,
        "severity":       severity,
        "lesion_pixels":  lesion_pixels,
        "total_pixels":   total_pixels,
        "confidence":     confidence,
        "inference_time": inference_time,
        "original_size":  original_size,
        "prob_stats": {
            "min":  float(prob.min()),
            "max":  float(prob.max()),
            "mean": float(prob.mean()),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.25rem;">
        <div style="font-size:1rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.3px;">⬡ PsoriScan</div>
        <div style="font-size:0.7rem;color:#475569;letter-spacing:1px;text-transform:uppercase;margin-top:2px;">Research Tool · v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-label'>Model</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="model-info">
        arch&nbsp;&nbsp;&nbsp; <span>U-Net (3-level)</span><br>
        device&nbsp; <span>{DEVICE.upper()}</span><br>
        input&nbsp;&nbsp; <span>{IMG_SIZE}×{IMG_SIZE} px</span><br>
        file&nbsp;&nbsp;&nbsp; <span>{MODEL_FILENAME}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Detection</div>", unsafe_allow_html=True)
    threshold = st.slider(
        "Threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Sigmoid probability cutoff for lesion/background classification"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Visualisation</div>", unsafe_allow_html=True)
    show_mask      = st.checkbox("Binary mask",       value=True)
    show_overlay   = st.checkbox("Colour overlay",    value=True)
    show_blended   = st.checkbox("Blended view",      value=True)
    show_heatmap   = st.checkbox("Grad-CAM",          value=True)
    show_probmap   = st.checkbox("Probability map",   value=False)
    show_debug     = st.checkbox("Debug info",        value=False)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.73rem;color:#334155;line-height:1.7;">
        Automated segmentation using a trained U-Net model on ISIC 2018 data.
        For research and educational use only. Not a medical device.<br><br>
        Dice&nbsp;≈ 0.79 · IoU&nbsp;≈ 0.70 on 180-image test set.
    </div>
    """, unsafe_allow_html=True)


# ── Brand header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-wrap">
    <div class="brand-title">Psoriasis Lesion Analyser</div>
    <div class="brand-badge">Research</div>
</div>
<div class="brand-subtitle">AI-assisted segmentation and severity estimation using deep learning</div>
""", unsafe_allow_html=True)

# ── Load model (cached) ─────────────────────────────────────────────────────────
with st.spinner("Loading model weights…"):
    model = load_model()

# ── Upload ───────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a dermoscopy or clinical skin image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

# ── Main content ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner("Running segmentation…"):
        results = process_image(uploaded_file, model, threshold)

    sev_level, sev_class, sev_icon = get_severity_level(results["severity"])

    # ── Result header bar ───────────────────────────────────────────────────────
    ts_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    st.markdown(f"""
    <div class="result-header">
        <div class="result-header-left">
            <div class="result-header-title">Analysis complete</div>
            <div class="result-header-file">{uploaded_file.name}</div>
            <div class="result-header-ts">{ts_str}</div>
        </div>
        <span class="sev-pill {sev_class}">{sev_icon} {sev_level} severity</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row (4 Streamlit metric widgets) ────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Affected area",    f"{results['severity']}%",
                  help="Lesion pixels ÷ total pixels × 100")
    with c2:
        st.metric("Model confidence", f"{results['confidence']}%",
                  help="Mean sigmoid probability over detected lesion pixels. Replaces the previous Dice metric that incorrectly compared the mask to itself.")
    with c3:
        st.metric("Lesion pixels",    f"{results['lesion_pixels']:,}",
                  help=f"Out of {results['total_pixels']:,} total pixels")
    with c4:
        st.metric("Inference time",   f"{results['inference_time']:.3f}s",
                  help="Time from tensor creation to sigmoid output")

    # ── Debug panel ─────────────────────────────────────────────────────────────
    if show_debug:
        with st.expander("Debug — raw probability statistics"):
            s = results["prob_stats"]
            st.markdown(f"""
            <div class="debug-row">
                prob min   → {s['min']:.4f}<br>
                prob max   → {s['max']:.4f}<br>
                prob mean  → {s['mean']:.4f}<br>
                threshold  → {threshold}<br>
                lesion px  → {results['lesion_pixels']:,} / {results['total_pixels']:,}<br>
                orig size  → {results['original_size'][1]} × {results['original_size'][0]} px
            </div>
            """, unsafe_allow_html=True)

    # ── Visualisation tabs ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    tab_cmp, tab_det, tab_rep = st.tabs(["Comparison", "Detailed views", "Report"])

    with tab_cmp:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("<div class='img-label'>Original image</div>", unsafe_allow_html=True)
            st.image(results["original"], width="stretch")
        with col_r:
            if show_blended:
                st.markdown("<div class='img-label'>Segmentation — blended</div>", unsafe_allow_html=True)
                st.image(results["blended"], width="stretch")
            elif show_overlay:
                st.markdown("<div class='img-label'>Segmentation — overlay</div>", unsafe_allow_html=True)
                st.image(results["overlay"], width="stretch")
            else:
                st.markdown("<div class='img-label'>Binary mask</div>", unsafe_allow_html=True)
                st.image(results["mask"], clamp=True, width="stretch")

    with tab_det:
        cols = st.columns(2)
        idx  = 0
        panels = []
        if show_mask:
            panels.append(("Binary mask",         results["mask"],    True,  "White = lesion · Black = healthy"))
        if show_overlay:
            panels.append(("Colour overlay",      results["overlay"], False, "Highlighted lesion region"))
        if show_blended:
            panels.append(("Blended view",        results["blended"], False, "Semi-transparent lesion highlight"))
        if show_heatmap:
            panels.append(("Grad-CAM attention",  results["heatmap"], False, "Spatial attention — warmer = higher contribution"))
        if show_probmap:
            panels.append(("Probability map",     results["prob_map"],True,  f"Sigmoid output · range [{results['prob_stats']['min']:.2f}–{results['prob_stats']['max']:.2f}]"))

        for title, img, clamp, caption in panels:
            with cols[idx % 2]:
                st.markdown(f"<div class='img-label'>{title}</div>", unsafe_allow_html=True)
                if clamp:
                    st.image(img, clamp=True, width="stretch")
                else:
                    st.image(img, width="stretch")
                st.caption(caption)
            idx += 1

    with tab_rep:
        orig_h, orig_w = results["original_size"]
        sev_pct = results["severity"]
        conf    = results["confidence"]
        inf_t   = results["inference_time"]

        st.markdown(f"""
**File:** `{uploaded_file.name}`  
**Analysed:** {ts_str}  
**Original dimensions:** {orig_w} × {orig_h} px → resized to {IMG_SIZE}×{IMG_SIZE} px

---

#### Segmentation
| Metric | Value |
|--------|-------|
| Severity score | **{sev_pct}%** |
| Severity level | **{sev_level}** |
| Lesion pixels | **{results['lesion_pixels']:,}** of {results['total_pixels']:,} |
| Detection threshold | `{threshold}` |

#### Model output
| Metric | Value |
|--------|-------|
| Model confidence (lesion px) | **{conf}%** |
| Prob. range | {results['prob_stats']['min']:.3f} – {results['prob_stats']['max']:.3f} |
| Inference time | {inf_t:.3f} s |
| Device | `{DEVICE.upper()}` |
| Architecture | U-Net (3-level encoder, BatchNorm, Dropout2d) |
        """)

        if sev_pct < 10:
            st.success("Low severity — minimal lesion coverage detected.")
        elif sev_pct < 30:
            st.warning("Moderate severity — significant lesion coverage detected.")
        else:
            st.error("High severity — extensive lesion coverage detected.")

        st.markdown("""
<div class="callout">
⚠️ &nbsp;This is an automated research tool.
Results must be reviewed by a qualified medical professional.
This software does not constitute a clinical diagnosis.
</div>
""", unsafe_allow_html=True)

else:
    # ── Welcome / idle state ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="welcome-panel">
        <h2>Upload an image to begin</h2>
        <p>The model will segment psoriasis lesions, estimate severity,
        and visualise model attention using Grad-CAM.</p>
        <div class="steps-row">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-text">Upload image</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-text">Adjust threshold</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-text">Review results</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid #1e293b;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
    <span style="font-size:0.73rem;color:#334155;">
        Psoriasis Lesion Analyser &nbsp;·&nbsp; U-Net · ISIC 2018 · Research use only
    </span>
    <span style="font-size:0.73rem;color:#334155;">
        Avg Dice 0.79 &nbsp;·&nbsp; Avg IoU 0.70 &nbsp;·&nbsp; 1 200-image training set
    </span>
</div>
""", unsafe_allow_html=True)
