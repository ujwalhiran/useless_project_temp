# 🍌 RipenAI - Deep Learning & Computer Vision Banana Ripeness Classifier

A hybrid Deep Learning (MobileNetV2 CNN) and Computer Vision system that trains directly on banana image datasets to predict ripeness, classify commercial ripening stages, and estimate shelf-life.

---

## 🚀 Quick Start (Hackathon Setup)

### 1. Install Dependencies
In `e:\hackathon1`:
```bash
pip install -r requirements.txt
```

### 2. Download Image Dataset & Train CNN
```bash
# Step A: Download curated banana image dataset
python dataset_setup.py

# Step B: Train the MobileNetV2 Neural Network on image data
python train_cnn.py
```

### 3. Launch the Web App
```bash
streamlit run app.py
```
Or open in Jupyter:
```bash
jupyter notebook banana_ripeness_analysis.ipynb
```

---

## 🧠 Deep Learning Architecture

### 1. Image Dataset Pipeline (`dataset/`)
Image data is organized into standard PyTorch ImageFolder format:
```
dataset/
├── train/
│   ├── unripe/       # Photos of green, starchy bananas
│   ├── ripe/         # Photos of yellow, sweet bananas
│   └── overripe/     # Photos of spotted, softening bananas
└── val/
    ├── unripe/
    ├── ripe/
    └── overripe/
```
You can drop any extra real photos into these folders anytime!

### 2. Transfer Learning with MobileNetV2
- **Pre-trained Backbone**: MobileNetV2 trained on ImageNet (extracts edges, surface textures, freckling patterns).
- **Custom Classification Head**: Dropout(0.3) ➔ Linear(128) ➔ ReLU ➔ Linear(3 classes).
- **Softmax Probabilities**: Returns confidence score for each class (`Unripe`, `Ripe`, `Overripe`).

---

## 📁 Project Structure

```
e:\hackathon1\
├── app.py                         # Streamlit Dashboard (Hybrid CNN + CV)
├── dataset_setup.py               # Downloads and prepares image data folders
├── train_cnn.py                   # Trains MobileNetV2 on image dataset
├── cnn_detector.py                # Deep learning inference on raw image tensors
├── ripeness_detector.py           # Spectral color engine & shelf-life calculator
├── generate_samples.py            # Generates test images
├── banana_ripeness_analysis.ipynb # Jupyter notebook with full DL pipeline
└── requirements.txt               # Dependencies (torch, torchvision, streamlit, etc.)
```
