import streamlit as st
import torch
import cv2
import numpy as np

from segmentation_model import UNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "project/unet_model.pth"


@st.cache_resource
def load_model():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


model = load_model()

st.title("Psoriasis Lesion Segmentation ")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    input_img = img / 255.0
    tensor = np.transpose(input_img, (2, 0, 1))
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # === Segmentation ===
    with torch.no_grad():
        pred = model(tensor)[0, 0].cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8)

    lesion_pixels = np.sum(mask)
    total_pixels = mask.size
    severity = round((lesion_pixels / total_pixels) * 100, 2)

    overlay = img.copy()
    overlay[mask == 1] = [255, 0, 0]

    # === Grad-CAM ===
    target_layers = [model.conv1.conv[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    def segmentation_target(output):
        return output.mean()

    grayscale_cam = cam(input_tensor=tensor, targets=[segmentation_target])[0]
    heatmap = show_cam_on_image(input_img, grayscale_cam, use_rgb=True)

    # === Display ===
    st.subheader(f"Severity Score: {severity}%")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original Image")

    with col2:
        st.image(overlay, caption="Segmentation Overlay")

    st.image(heatmap, caption="Grad-CAM Explainability")
