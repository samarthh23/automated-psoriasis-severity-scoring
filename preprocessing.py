import cv2
import os
import matplotlib.pyplot as plt
import config

IMAGE_DIR = config.IMAGE_DIR
MASK_DIR = config.MASK_DIR
IMG_SIZE = config.IMG_SIZE


def load_pair():
    images = os.listdir(IMAGE_DIR)
    if not images:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}. Run dataset_prepare.py first.")
    img_name = images[0]
    base = os.path.splitext(img_name)[0]  # handles .jpg, .jpeg, .png, etc.
    mask_name = base + "_segmentation.png"

    img_path = os.path.join(IMAGE_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, mask_name)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    return image, mask


def visualize():
    image, mask = load_pair()

    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]  # red lesion

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    visualize()
