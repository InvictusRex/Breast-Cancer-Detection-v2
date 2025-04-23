import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class
class ClassificationDataset(Dataset):
    def __init__(self, image_dir, transform, label_map):
        self.image_files = []
        self.labels = []
        self.transform = transform

        for label, folder in label_map.items():
            folder_path = os.path.join(image_dir, folder)
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
            self.image_files.extend(files)
            self.labels.extend([label] * len(files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return img, label

# Model definition
class SimCLRClassifier(nn.Module):
    def __init__(self):
        super(SimCLRClassifier, self).__init__()
        self.encoder = models.resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

# Transforms
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths and loader
data_dir = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\flattened_split'
label_map = {0: "benign", 1: "malignant"}
test_dataset = ClassificationDataset(os.path.join(data_dir, "test"), transform, label_map)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Load model and weights
model = SimCLRClassifier().to(device)
model.load_state_dict(torch.load("best_finetuned_model_resnet50.pth", map_location=device))
model.eval()

# Evaluation function
def evaluate(model, loader, verbose=True):
    model.eval()
    total_loss, y_true, y_pred, y_probs = 0, [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(y_true, y_pred)

    if verbose:
        print("📊 Final Test Evaluation")
        print(f"Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        print("Precision:", precision_score(y_true, y_pred))
        print("Recall:", recall_score(y_true, y_pred))
        print("F1 Score:", f1_score(y_true, y_pred))
        print("AUC-ROC:", roc_auc_score(y_true, y_probs))
        print("Classification Report:\n", classification_report(y_true, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Sample predictions
        visualize_predictions(model, loader)

    return avg_loss, acc

# Visualization
def visualize_predictions(model, loader, num_images=8):
    model.eval()
    images, labels, preds = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            output = model(x)
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).float().cpu()
            preds.extend(pred.squeeze().tolist())
            labels.extend(y.squeeze().tolist())
            images.extend(x.cpu())
            if len(images) >= num_images:
                break

    plt.figure(figsize=(16, 6))
    for i in range(num_images):
        img = images[i].permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # unnormalize
        img = np.clip(img, 0, 1)
        plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"True: {int(labels[i])}, Pred: {int(preds[i])}")
    plt.suptitle("Sample Predictions")
    plt.show()

# Run evaluation
evaluate(model, test_loader, verbose=True)
