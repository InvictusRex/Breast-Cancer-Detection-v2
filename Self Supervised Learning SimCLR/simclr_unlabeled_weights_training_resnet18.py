import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True

# --- Transformations ---
def get_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Dataset ---
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        img1, img2 = self.transform(img), self.transform(img)
        return img1, img2, img_path  # return img_path for augmented visualization

# --- Model ---
class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return projections

# --- Loss ---
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T)
    logits = sim_matrix / temperature
    labels = torch.arange(2 * batch_size, dtype=torch.long, device=device)
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits, labels)

# --- Training ---
def train(model, train_loader, val_loader, optimizer, num_epochs=100, patience=15):
    best_loss = float('inf')
    best_model_path = "best_simclr_model.pth"
    patience_counter = 0

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for x1, x2, _ in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            z1, z2 = model(x1), model(x2)
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x1, x2, _ in val_loader:
                x1, x2 = x1.to(device), x2.to(device)
                z1, z2 = model(x1), model(x2)
                loss = nt_xent_loss(z1, z2)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # --- Visualizations ---
    plot_loss(train_losses, val_losses)
    visualize_tsne(model, val_loader)
    visualize_augmentations(val_loader)

# --- Plot Train vs Val Loss ---
def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SimCLR Training Loss')
    plt.legend()
    plt.show()

# --- t-SNE Embedding ---
def visualize_tsne(model, dataloader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for x1, _, _ in dataloader:
            x1 = x1.to(device)
            z = model.encoder(x1)
            embeddings.append(z.cpu().numpy())
            if len(embeddings) * x1.size(0) >= 300:  # limit to ~300 samples
                break

    embeddings = np.concatenate(embeddings, axis=0)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced = tsne.fit_transform(embeddings)

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], s=15, alpha=0.7)
    plt.title('t-SNE Projection of Learned Embeddings')
    plt.show()

# --- Augmented Views Visualization ---
def visualize_augmentations(dataloader, num_images=5):
    to_pil = transforms.ToPILImage()
    images_shown = 0
    for x1, x2, paths in dataloader:
        for i in range(len(x1)):
            if images_shown >= num_images:
                return
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(to_pil(x1[i].cpu()))
            axs[0].set_title("Augmented View 1")
            axs[1].imshow(to_pil(x2[i].cpu()))
            axs[1].set_title("Augmented View 2")
            plt.suptitle(os.path.basename(paths[i]))
            plt.show()
            images_shown += 1

# --- Paths ---
train_dir = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\unlabeled_split\train'
val_dir = r'E:\1_Work_Files\6_Project - Breast Cancer Detection\BreaKHis_v1\unlabeled_split\val'

transform = get_transforms()
train_dataset = ImageDataset(train_dir, transform)
val_dataset = ImageDataset(val_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False, num_workers=0)

model = SimCLR().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, val_loader, optimizer, num_epochs=100, patience=15)
