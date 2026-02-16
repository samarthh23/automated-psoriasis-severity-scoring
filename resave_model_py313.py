"""
Quick Model Re-save Script for Python 3.13 Compatibility
This script loads the existing model and re-saves it in a format compatible with Python 3.13
"""

import torch
from segmentation_model import UNet
import config

print("=" * 70)
print("MODEL RE-SAVE FOR PYTHON 3.13 COMPATIBILITY")
print("=" * 70)

# Load the existing model
print("\n1. Loading existing model...")
model = UNet().to(config.DEVICE)

try:
    # Try loading with weights_only=False
    state_dict = torch.load(
        "models/best_unet_model.pth", map_location=config.DEVICE, weights_only=False
    )
    model.load_state_dict(state_dict)
    print("   ✓ Model loaded successfully")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    print("\n   Trying alternative model file...")
    try:
        state_dict = torch.load(
            "unet_model.pth", map_location=config.DEVICE, weights_only=False
        )
        model.load_state_dict(state_dict)
        print("   ✓ Model loaded from unet_model.pth")
    except Exception as e2:
        print(f"   ✗ Error: {e2}")
        exit(1)

# Re-save with protocol 4 (compatible with Python 3.4+, including 3.13)
print("\n2. Re-saving model with pickle protocol 4...")
torch.save(model.state_dict(), "models/unet_model_py313.pth", pickle_protocol=4)
print("   ✓ Saved to: models/unet_model_py313.pth")

# Also save to the default location
torch.save(model.state_dict(), "models/unet_model.pth", pickle_protocol=4)
print("   ✓ Saved to: models/unet_model.pth")

# Save to root as well
torch.save(model.state_dict(), "unet_model.pth", pickle_protocol=4)
print("   ✓ Saved to: unet_model.pth")

print("\n" + "=" * 70)
print("✓ MODEL RE-SAVED SUCCESSFULLY!")
print("=" * 70)
print("\nNext steps:")
print("1. Test locally: streamlit run app.py")
print("2. Commit and push:")
print("   git add models/unet_model.pth unet_model.pth")
print("   git commit -m 'Re-save model for Python 3.13 compatibility'")
print("   git push")
print("\n" + "=" * 70)
