# Streamlit Cloud Deployment Guide

## 🚀 Quick Deployment Steps

### 1. Prepare Repository (Already Done ✓)

- `.gitignore` configured to exclude datasets, models, and logs
- Only essential code files will be pushed

### 2. Add Remote Repository

```bash
git remote add origin https://github.com/samarthh23/automated-psoriasis-severity-scoring.git
```

### 3. Stage and Commit Changes

```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
```

### 4. Push to GitHub

```bash
git push -u origin master
```

### 5. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select repository: `samarthh23/automated-psoriasis-severity-scoring`
5. Branch: `master`
6. Main file path: `app.py`
7. Click **"Deploy"**

---

## ⚠️ Important: Model File Handling

Since the model file (`unet_model.pth` - 7.4MB) is excluded from Git, you need to handle it separately:

### Option 1: Download Model on First Run (Recommended)

Add this to your `app.py` before loading the model:

```python
import os
import urllib.request

MODEL_URL = "YOUR_MODEL_URL_HERE"  # Upload to Google Drive, Dropbox, or Hugging Face
MODEL_PATH = "models/unet_model.pth"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    with st.spinner("Downloading model... (one-time setup)"):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
```

### Option 2: Use Git LFS (Large File Storage)

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add models/unet_model.pth
git commit -m "Add model with Git LFS"
git push
```

### Option 3: Hugging Face Hub (Best for ML Models)

```bash
pip install huggingface-hub
```

Upload model:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="models/unet_model.pth",
    path_in_repo="unet_model.pth",
    repo_id="samarthh23/psoriasis-segmentation",
    repo_type="model"
)
```

Download in app:

```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="samarthh23/psoriasis-segmentation",
    filename="unet_model.pth"
)
```

---

## 📦 Files That Will Be Pushed

### ✅ Included (Essential Files)

- `app.py` - Main Streamlit application
- `config.py` - Configuration settings
- `segmentation_model.py` - U-Net model architecture
- `severity_model.py` - Severity scoring module
- `explainability.py` - Grad-CAM visualization
- `dataset.py` - Dataset handling
- `preprocessing.py` - Image preprocessing
- `train_segmentation.py` - Training script
- `train_with_validation.py` - Training with validation
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `USAGE_GUIDE.md` - Usage instructions

### ❌ Excluded (Not Needed for Deployment)

- `data/` - Dataset images and masks (large files)
- `models/` - Trained model files (7.4MB)
- `logs/` - Training logs and plots
- `__pycache__/` - Python cache
- `.streamlit/` - Local Streamlit config
- `.agent/` - AI agent workflows
- Internal documentation (APP_IMPROVEMENTS.md, BUGFIX_SIGMOID.md, etc.)

---

## 🔧 Configuration for Streamlit Cloud

### Update `config.py` for Cloud Deployment

Ensure your config works without GPU:

```python
import torch

# Auto-detect device (Streamlit Cloud doesn't have GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Create `.streamlit/config.toml` (Optional)

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 10
enableXsrfProtection = true
```

---

## 🧪 Testing Before Deployment

### Test Locally

```bash
streamlit run app.py
```

### Verify Requirements

```bash
pip install -r requirements.txt
```

### Check Model Loading

```bash
python -c "from segmentation_model import UNet; print('Model architecture OK')"
```

---

## 🐛 Troubleshooting

### Issue: "Model file not found"

**Solution**: Implement one of the model handling options above

### Issue: "Module not found"

**Solution**: Ensure all dependencies are in `requirements.txt`

### Issue: "Memory error on Streamlit Cloud"

**Solution**: Streamlit Cloud has 1GB RAM limit. Optimize model loading:

```python
@st.cache_resource
def load_model():
    model = UNet().to("cpu")  # Force CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return model
```

### Issue: "Deployment fails"

**Solution**: Check Streamlit Cloud logs for specific errors

---

## 📊 Post-Deployment Checklist

- [ ] Repository pushed to GitHub
- [ ] App deployed on Streamlit Cloud
- [ ] Model file accessible (via download or Git LFS)
- [ ] App loads without errors
- [ ] Test image upload functionality
- [ ] Verify segmentation results
- [ ] Check Grad-CAM visualization
- [ ] Share public URL with users

---

## 🔗 Useful Links

- **Streamlit Cloud**: https://share.streamlit.io
- **Streamlit Docs**: https://docs.streamlit.io
- **Git LFS**: https://git-lfs.github.com
- **Hugging Face Hub**: https://huggingface.co/docs/hub

---

## 📝 Next Steps After Deployment

1. **Get your app URL**: `https://share.streamlit.io/samarthh23/automated-psoriasis-severity-scoring/master/app.py`
2. **Monitor usage**: Check Streamlit Cloud dashboard for analytics
3. **Update app**: Push changes to GitHub, Streamlit Cloud auto-deploys
4. **Share**: Send the URL to users or embed in documentation

---

**Need Help?** Check the Streamlit Community Forum or GitHub Issues.
