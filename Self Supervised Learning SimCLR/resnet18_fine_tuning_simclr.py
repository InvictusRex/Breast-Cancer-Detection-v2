import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
import seaborn as sns

# Device config
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

# Paths & settings
data_dir = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\flattened_split'
label_map = {0: "benign", 1: "malignant"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ClassificationDataset(os.path.join(data_dir, "train"), transform, label_map)
val_dataset   = ClassificationDataset(os.path.join(data_dir, "val"), transform, label_map)
test_dataset  = ClassificationDataset(os.path.join(data_dir, "test"), transform, label_map)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=512, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True)

# Model definition
class SimCLRClassifier(nn.Module):
    def __init__(self):
        super(SimCLRClassifier, self).__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

model = SimCLRClassifier().to(device)

# Load pretrained encoder weights
pretrained_weights = torch.load(r'E:\1_Work_Files\6_Project - Breast Cancer Detection\Breast-Cancer-Detection-v2\Models\best_simclr_model.pth', map_location=device)
model.encoder.load_state_dict(pretrained_weights, strict=False)

# Freeze encoder except last block
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.encoder.layer4.parameters():
    param.requires_grad = True

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

# Training
def train(model, train_loader, val_loader, num_epochs=100, patience=15):
    best_loss = float("inf")
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == y).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = total_correct / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, verbose=False)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} → Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_finetuned_model.pth")
            print(f"✅ Best model saved with val loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    plot_metrics(train_losses, val_losses, train_accs, val_accs)

# Evaluation
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

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        visualize_predictions(model, test_loader)

    return avg_loss, acc

# Plotting
def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Visualize Predictions
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

# Train the model
train(model, train_loader, val_loader, num_epochs=100, patience=15)

# Load best and evaluate
model.load_state_dict(torch.load("best_finetuned_model.pth"))
evaluate(model, test_loader, verbose=True)
