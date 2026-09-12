"""
CNN Deep Learning Inference Module
Loads trained MobileNetV2 weights and performs inference on raw image data.
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class DeepBananaClassifier:
    def __init__(self, model_path="e:/hackathon1/banana_model.pth", classes_path="e:/hackathon1/classes.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.classes_path = classes_path
        self.class_names = ["overripe", "ripe", "unripe"] # default fallback
        
        if os.path.exists(classes_path):
            try:
                with open(classes_path, "r") as f:
                    self.class_names = json.load(f)
            except Exception:
                pass

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = self._load_model()

    def _load_model(self):
        # Initialize MobileNetV2 architecture
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.class_names))
        )

        if os.path.exists(self.model_path):
            try:
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(" Loaded custom trained banana weights from:", self.model_path)
            except Exception as e:
                print("⚠️ Could not load saved weights, using base architecture:", e)
        else:
            print("ℹ️ Weights file not found. Model is ready to be trained using 'python train_cnn.py'")

        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, image_input):
        """
        Accepts PIL Image, path, or numpy array.
        Returns predicted class, confidence, and probability distribution across all classes.
        """
        if isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, str):
            pil_img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
        else:
            raise ValueError("Unsupported image type for CNN inference")

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        predicted_idx = int(np.argmax(probabilities))
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx]) * 100

        prob_dict = {
            self.class_names[i].capitalize(): round(float(probabilities[i]) * 100, 1)
            for i in range(len(self.class_names))
        }

        # Calculate an overall continuous ripeness index from probabilities
        # Unripe = ~20%, Ripe = ~80%, Overripe = ~100%
        ripeness_score = 0
        if "Unripe" in prob_dict:
            ripeness_score += prob_dict["Unripe"] * 0.2
        if "Ripe" in prob_dict:
            ripeness_score += prob_dict["Ripe"] * 0.85
        if "Overripe" in prob_dict:
            ripeness_score += prob_dict["Overripe"] * 1.0

        return {
            "predicted_class": predicted_class.capitalize(),
            "confidence": round(confidence, 1),
            "probabilities": prob_dict,
            "deep_ripeness_score": min(round(ripeness_score, 1), 100.0)
        }
