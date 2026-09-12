"""
High-Accuracy Deep Learning CNN Trainer for Banana Ripeness
Fine-tunes MobileNetV2 with unfrozen top conv layers and learning rate scheduler.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def train_banana_model(data_dir="e:/hackathon1/dataset", num_epochs=8, batch_size=8, lr=0.0005):
    print("🚀 Initializing High-Accuracy Deep Learning Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # Advanced Data Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        import dataset_setup
        dataset_setup.download_and_expand_dataset(data_dir)

    train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes
    print(f"Ripeness Classes: {class_names} | Total Training Images: {len(train_dataset)}")

    with open("e:/hackathon1/classes.json", "w") as f:
        json.dump(class_names, f)

    # Load MobileNetV2
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze earlier layers, but unfreeze the last 4 feature layers for texture adaptation
    for param in model.features[:-4].parameters():
        param.requires_grad = False
    for param in model.features[-4:].parameters():
        param.requires_grad = True

    # Custom Classification Head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(256, len(class_names))
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    print("\n🏋️ Training Deep Learning Model on 150+ Banana Samples...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (corrects.double() / len(train_dataset)) * 100

        # Quick validation evaluation
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for v_inputs, v_labels in val_loader:
                v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                v_out = model(v_inputs)
                _, v_preds = torch.max(v_out, 1)
                val_corrects += torch.sum(v_preds == v_labels.data)
        val_acc = (val_corrects.double() / len(val_dataset)) * 100

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    save_path = "e:/hackathon1/banana_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 High-Accuracy Model saved to '{save_path}'!")
    return save_path

if __name__ == "__main__":
    train_banana_model()
