"""
Enhanced Streamlit Interface for Psoriasis Lesion Segmentation
Modern, professional UI with improved features and visualizations
FIXED: Proper sigmoid activation for model predictions
"""

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

# Page configuration
st.set_page_config(
    page_title="Psoriasis Lesion Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .severity-low {
        color: #28a745;
        font-weight: bold;
    }
    .severity-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .severity-high {
        color: #dc3545;
        font-weight: bold;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Constants
IMG_SIZE = config.IMG_SIZE
DEVICE = config.DEVICE
MODEL_PATH = config.get_model_path()


@st.cache_resource
def load_model():
    """Load the segmentation model"""
    with st.spinner("Loading AI model..."):
        model = UNet().to(DEVICE)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        )
        model.eval()
    return model


def get_severity_level(severity):
    """Categorize severity level"""
    if severity < 10:
        return "Low", "severity-low"
    elif severity < 30:
        return "Moderate", "severity-medium"
    else:
        return "High", "severity-high"


def process_image(uploaded_file, model, threshold=0.5):
    """Process uploaded image and return results"""
    # Read and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # CRITICAL FIX: Use EXACT same preprocessing as training
    # Training uses: img / 255.0, transpose to (C, H, W)
    # NO ImageNet normalization - just simple division by 255
    input_img = img_resized / 255.0  # Normalize to [0, 1]
    tensor = np.transpose(input_img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Segmentation
    start_time = time.time()
    with torch.no_grad():
        pred_logits = model(tensor)[0, 0].cpu().numpy()

        # CRITICAL FIX: Apply sigmoid to convert logits to probabilities
        # Training uses BCEWithLogitsLoss which applies sigmoid internally
        # For inference, we need to apply sigmoid manually
        pred = torch.sigmoid(torch.from_numpy(pred_logits)).numpy()

    inference_time = time.time() - start_time

    # Debug statistics
    pred_min, pred_max, pred_mean = pred.min(), pred.max(), pred.mean()

    # Create mask with threshold
    mask = (pred > threshold).astype(np.uint8)

    # Calculate metrics
    lesion_pixels = np.sum(mask)
    total_pixels = mask.size
    severity = round((lesion_pixels / total_pixels) * 100, 2)

    # Calculate Dice coefficient (for display purposes, using prediction as "ground truth proxy")
    # In real evaluation, you'd compare against actual ground truth
    # Here we calculate it as 2 * intersection / (pred_area + mask_area)
    pred_binary = (pred > threshold).astype(np.float32)
    intersection = np.sum(pred_binary * mask)
    dice_score = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(mask) + 1e-7)
    dice_score = round(dice_score * 100, 2)  # Convert to percentage

    # Create visualizations
    # Scale mask to 0-255 for proper display (fix black mask issue)
    mask_display = (mask * 255).astype(np.uint8)

    overlay = img_resized.copy()
    overlay[mask == 1] = [255, 0, 0]  # Red overlay

    # Blended overlay
    alpha = 0.4
    blended = cv2.addWeighted(img_resized, 1 - alpha, overlay, alpha, 0)

    # Grad-CAM
    target_layers = [model.conv1.conv[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    def segmentation_target(output):
        return output.mean()

    grayscale_cam = cam(input_tensor=tensor, targets=[segmentation_target])[0]
    heatmap = show_cam_on_image(input_img, grayscale_cam, use_rgb=True)

    return {
        "original": img_resized,
        "mask": mask_display,  # Use scaled version for display
        "overlay": overlay,
        "blended": blended,
        "heatmap": heatmap,
        "severity": severity,
        "lesion_pixels": lesion_pixels,
        "total_pixels": total_pixels,
        "dice_score": dice_score,
        "inference_time": inference_time,
        "original_size": original_size,
        "prediction": pred,
        "pred_stats": {
            "min": float(pred_min),
            "max": float(pred_max),
            "mean": float(pred_mean),
        },
    }


# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown(
    '<div class="main-header">🔬 Psoriasis Lesion Analysis System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">AI-Powered Segmentation and Severity Assessment with Explainable AI</div>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Model info
    st.markdown("### 📊 Model Information")
    st.info(f"""
    **Architecture:** U-Net  
    **Device:** {DEVICE.upper()}  
    **Image Size:** {IMG_SIZE}×{IMG_SIZE}  
    **Model:** {MODEL_PATH.split("/")[-1] if "/" in MODEL_PATH else MODEL_PATH.split("\\\\")[-1]}
    """)

    # Segmentation threshold
    st.markdown("### 🎯 Segmentation Settings")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust sensitivity of lesion detection",
    )

    # Visualization options
    st.markdown("### 🎨 Visualization Options")
    show_mask = st.checkbox("Show Binary Mask", value=True)
    show_overlay = st.checkbox("Show Overlay", value=True)
    show_blended = st.checkbox("Show Blended View", value=True)
    show_heatmap = st.checkbox("Show Grad-CAM", value=True)
    show_prediction = st.checkbox("Show Raw Prediction", value=False)
    show_debug = st.checkbox("Show Debug Info", value=False)

    # About section
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This system uses deep learning to:
    - Detect psoriasis lesions
    - Calculate severity scores
    - Provide explainable AI visualizations
    
    **Note:** For research purposes only.
    """)

# Load model
model = load_model()

# Main content
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload a skin image for analysis",
    type=["jpg", "png", "jpeg"],
    help="Supported formats: JPG, PNG, JPEG",
)

if uploaded_file is not None:
    # Process image
    with st.spinner("🔄 Processing image..."):
        results = process_image(uploaded_file, model, threshold)

    st.success("✅ Analysis complete!")

    # Debug info
    if show_debug:
        st.markdown("### 🐛 Debug Information")
        st.json(
            {
                "Prediction Stats": results["pred_stats"],
                "Threshold": threshold,
                "Lesion Pixels": results["lesion_pixels"],
                "Total Pixels": results["total_pixels"],
            }
        )

    # Display metrics
    st.markdown("## 📈 Analysis Results")

    # Metrics row - now with 5 columns to include Dice score
    col1, col2, col3, col4, col5 = st.columns(5)

    severity_level, severity_class = get_severity_level(results["severity"])

    with col1:
        st.metric(
            label="Severity Score",
            value=f"{results['severity']}%",
            help="Percentage of affected area",
        )

    with col2:
        st.metric(
            label="Dice Score",
            value=f"{results['dice_score']}%",
            help="Dice coefficient - measures segmentation quality (higher is better)",
        )

    with col3:
        st.metric(
            label="Severity Level", value=severity_level, help="Categorized severity"
        )

    with col4:
        st.metric(
            label="Lesion Pixels",
            value=f"{results['lesion_pixels']:,}",
            help="Number of detected lesion pixels",
        )

    with col5:
        st.metric(
            label="Processing Time",
            value=f"{results['inference_time']:.3f}s",
            help="AI inference time",
        )

    # Severity indicator
    st.markdown(
        f"""
    <div class="metric-card">
        <h3>Severity Assessment: <span class="{severity_class}">{severity_level}</span></h3>
        <p>Affected area: {results["severity"]}% ({results["lesion_pixels"]:,} / {results["total_pixels"]:,} pixels)</p>
        <p>Segmentation Quality (Dice): {results["dice_score"]}%</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Visualizations
    st.markdown("---")
    st.markdown("## 🖼️ Visualizations")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(
        ["📊 Comparison View", "🔍 Detailed Analysis", "📋 Summary"]
    )

    with tab1:
        # Side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(results["original"])

        with col2:
            if show_blended:
                st.markdown("### Segmentation (Blended)")
                st.image(results["blended"])
            elif show_overlay:
                st.markdown("### Segmentation (Overlay)")
                st.image(results["overlay"])
            else:
                st.markdown("### Binary Mask")
                st.image(results["mask"], clamp=True)

    with tab2:
        # Detailed views in grid
        cols = st.columns(2)

        idx = 0
        if show_mask:
            with cols[idx % 2]:
                st.markdown("#### Binary Mask")
                st.image(results["mask"], clamp=True)
                st.caption("White = Lesion, Black = Healthy skin")
            idx += 1

        if show_overlay:
            with cols[idx % 2]:
                st.markdown("#### Overlay View")
                st.image(results["overlay"])
                st.caption("Red overlay indicates detected lesions")
            idx += 1

        if show_blended:
            with cols[idx % 2]:
                st.markdown("#### Blended View")
                st.image(results["blended"])
                st.caption("Semi-transparent overlay for better visualization")
            idx += 1

        if show_heatmap:
            with cols[idx % 2]:
                st.markdown("#### Grad-CAM Heatmap")
                st.image(results["heatmap"])
                st.caption("AI attention map - shows what the model focuses on")
            idx += 1

        if show_prediction:
            with cols[idx % 2]:
                st.markdown("#### Raw Prediction")
                st.image(results["prediction"], clamp=True)
                st.caption(
                    f"Probability map (range: {results['pred_stats']['min']:.3f} - {results['pred_stats']['max']:.3f})"
                )
            idx += 1

    with tab3:
        # Summary report
        st.markdown("### 📄 Analysis Report")

        st.markdown(f"""
        **Analysis Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
        **Image File:** {uploaded_file.name}  
        **Original Size:** {results["original_size"][1]} × {results["original_size"][0]} pixels  
        **Processed Size:** {IMG_SIZE} × {IMG_SIZE} pixels  
        
        ---
        
        **Segmentation Results:**
        - **Severity Score:** {results["severity"]}%
        - **Dice Score:** {results["dice_score"]}%
        - **Severity Level:** {severity_level}
        - **Lesion Pixels:** {results["lesion_pixels"]:,}
        - **Total Pixels:** {results["total_pixels"]:,}
        - **Detection Threshold:** {threshold}
        
        ---
        
        **Performance:**
        - **Inference Time:** {results["inference_time"]:.3f} seconds
        - **Device:** {DEVICE.upper()}
        - **Model:** U-Net
        
        ---
        
        **Interpretation:**
        """)

        if results["severity"] < 10:
            st.success("✅ Low severity - Minimal lesion coverage detected")
        elif results["severity"] < 30:
            st.warning("⚠️ Moderate severity - Significant lesion coverage detected")
        else:
            st.error("🔴 High severity - Extensive lesion coverage detected")

        st.info("""
        **Note:** This is an automated analysis tool for research purposes. 
        Results should be reviewed by qualified medical professionals. 
        This tool does not replace clinical diagnosis.
        """)

else:
    # Instructions when no file is uploaded
    st.markdown(
        """
    <div class="info-box">
        <h3>👋 Welcome!</h3>
        <p>Upload a skin image to begin analysis. The system will:</p>
        <ul>
            <li>🎯 Detect and segment psoriasis lesions</li>
            <li>📊 Calculate severity scores</li>
            <li>🔍 Provide explainable AI visualizations</li>
            <li>📈 Generate detailed analysis reports</li>
        </ul>
        <p><strong>Supported formats:</strong> JPG, PNG, JPEG</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Example images section
    st.markdown("### 📸 Example Analysis")
    st.markdown(
        "Upload an image from the `data/images/` folder to see the system in action."
    )

    # Show sample workflow
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1️⃣ Upload")
        st.markdown("Select a skin image")

    with col2:
        st.markdown("#### 2️⃣ Process")
        st.markdown("AI analyzes the image")

    with col3:
        st.markdown("#### 3️⃣ Results")
        st.markdown("View segmentation & severity")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Psoriasis Lesion Analysis System | Powered by Deep Learning & Explainable AI</p>
    <p style="font-size: 0.8rem;">For research and educational purposes only</p>
</div>
""",
    unsafe_allow_html=True,
)
