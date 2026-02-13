import os
import shutil
import random

# ===== CHANGE THESE PATHS =====
SOURCE_IMAGES = r"F:\archive\ISIC2018_Task1-2_Training_Input"
SOURCE_MASKS = r"F:\archive\ISIC2018_Task1_Training_GroundTruth"

TARGET_IMAGES = r"C:\Users\Samarth Hegde\Desktop\project\data\images"
TARGET_MASKS = r"C:\Users\Samarth Hegde\Desktop\project\data\masks"

SAMPLE_SIZE = 80


def prepare_dataset():
    os.makedirs(TARGET_IMAGES, exist_ok=True)
    os.makedirs(TARGET_MASKS, exist_ok=True)

    image_files = [
        f for f in os.listdir(SOURCE_IMAGES)
        if f.endswith(".jpg")
    ]

    selected = random.sample(image_files, SAMPLE_SIZE)

    for img_name in selected:
        base = img_name.replace(".jpg", "")
        mask_name = base + "_segmentation.png"

        src_img = os.path.join(SOURCE_IMAGES, img_name)
        src_mask = os.path.join(SOURCE_MASKS, mask_name)

        if os.path.exists(src_mask):
            shutil.copy(src_img, os.path.join(TARGET_IMAGES, img_name))
            shutil.copy(src_mask, os.path.join(TARGET_MASKS, mask_name))

    print("Dataset prepared successfully!")


if __name__ == "__main__":
    prepare_dataset()
