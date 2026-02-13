import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segmentation_model import UNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_PATH = "unet_model.pth"
IMAGE_PATH = "data/images/" + sorted(__import__("os").listdir("data/images"))[0]

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load model
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# Preprocess image
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

input_img = img / 255.0
tensor = np.transpose(input_img, (2, 0, 1))
tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)


# Choose a convolution layer
target_layers = [model.conv1.conv[-1]]

cam = GradCAM(model=model, target_layers=target_layers)


# 🔥 KEY FIX: segmentation target
def segmentation_target(output):
    return output.mean()


grayscale_cam = cam(input_tensor=tensor, targets=[segmentation_target])[0]

visualization = show_cam_on_image(input_img, grayscale_cam, use_rgb=True)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.show()
