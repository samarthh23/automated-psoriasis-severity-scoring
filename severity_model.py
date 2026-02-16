import torch
import cv2
import numpy as np

import config
from segmentation_model import UNet

MODEL_PATH = config.get_model_path()
IMG_SIZE = config.IMG_SIZE
DEVICE = config.DEVICE


class SeverityScorer:
    def __init__(self):
        self.model = UNet().to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()

    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        normalized = img / 255.0
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

        return img, tensor.to(DEVICE)

    def predict_mask(self, tensor):
        with torch.no_grad():
            pred_logits = self.model(tensor)[0, 0].cpu().numpy()

            # Apply sigmoid to convert logits to probabilities
            # Model outputs raw logits, training uses BCEWithLogitsLoss
            pred = torch.sigmoid(torch.from_numpy(pred_logits)).numpy()

        binary_mask = (pred > 0.5).astype(np.uint8)
        return binary_mask

    def compute_severity(self, mask):
        lesion_pixels = np.sum(mask)
        total_pixels = mask.size

        severity = (lesion_pixels / total_pixels) * 100
        return round(severity, 2)

    def analyze(self, image_path):
        image, tensor = self.preprocess(image_path)
        mask = self.predict_mask(tensor)
        severity = self.compute_severity(mask)

        return image, mask, severity


if __name__ == "__main__":
    # Test on one training image
    test_image = "data/images/" + sorted(__import__("os").listdir("data/images"))[0]

    scorer = SeverityScorer()
    image, mask, severity = scorer.analyze(test_image)

    print("Severity Score:", severity, "%")

    overlay = image.copy()
    overlay[mask == 1] = [255, 0, 0]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Severity: {severity}%")
    plt.axis("off")

    plt.show()
