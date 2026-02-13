import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from segmentation_model import UNet

IMAGE_DIR = r"C:\Users\Samarth Hegde\Desktop\project\data\images"
MASK_DIR = r"C:\Users\Samarth Hegde\Desktop\project\data\masks"
IMG_SIZE = 256
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SkinDataset(Dataset):
    def __init__(self):
        self.images = os.listdir(IMAGE_DIR)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = img_name.replace(".jpg", "")
        mask_name = base + "_segmentation.png"

        img = cv2.imread(os.path.join(IMAGE_DIR, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))

        mask = cv2.imread(os.path.join(MASK_DIR, mask_name), 0)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


dataset = SkinDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "unet_model.pth")
print("Model saved!")


# === Visual Test ===
model.eval()
img, mask = dataset[0]
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(DEVICE)).cpu().numpy()[0, 0]

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
plt.title("Image")

plt.subplot(1, 3, 2)
plt.imshow(mask[0], cmap="gray")
plt.title("Ground Truth")

plt.subplot(1, 3, 3)
plt.imshow(pred, cmap="gray")
plt.title("Prediction")

plt.show()
