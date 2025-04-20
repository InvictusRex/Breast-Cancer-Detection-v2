# Breast Cancer Detection using Deep Learning (v2)
## A Comparative Study of Supervised and Self-Supervised Techniques on Histopathological Images for Breast Cancer Diagnosis using Deep Learning


## Overview
This project implements and compares three different deep learning approaches for breast cancer diagnosis using histopathological images. Using the BreaKHis dataset, we classify breast cancer images into benign and malignant categories through:

1. A basic Convolutional Neural Network (CNN) trained from scratch
2. Transfer learning with a pre-trained ResNet50 architecture
3. Self-supervised learning using SimCLR with ResNet18

## Objective
The primary objective is to explore and evaluate different deep learning strategies for automated classification of breast cancer histopathological images. We leverage the strengths of both traditional supervised learning and emerging self-supervised paradigms to design a robust diagnostic model, comparing their effectiveness in medical image analysis where labeled data is often limited.

## Dataset
We used the **BreaKHis** (Breast Cancer Histopathological Image Classification) dataset, which consists of microscopic images of breast tumor tissue collected at different magnification factors (40x, 100x, 200x, and 400x).

### Class Distribution
The dataset has an imbalanced distribution:
| Magnification | Benign | Malignant | Total | % Benign | % Malignant |
|---------------|--------|-----------|-------|----------|-------------|
| 40x           | 625    | 1370      | 1995  | 31.3%    | 68.7%       |
| 100x          | 644    | 1437      | 2081  | 30.9%    | 69.1%       |
| 200x          | 623    | 1390      | 2013  | 30.9%    | 69.1%       |
| 400x          | 588    | 1232      | 1820  | 32.3%    | 67.7%       |
| **Total**     | **2480** | **5429**  | **7909** | **31.4%** | **68.6%** |


## Project Structure
``` bash
.
├── Basic CNN Training/
│   ├── cnn_training_patience10.py         # CNN model with patience=10
│   ├── cnn_training_patience5.py          # CNN model with patience=5
│   └── dataset_partition.py               # Dataset splitting utilities
│
├── Transfer Learning on Pretrained Models/
│   ├── transfer_learning_mobilenet.py     # MobileNet implementation
│   ├── transfer_learning_resnet50.py      # ResNet50 implementation
│   └── transfer_learning_vgg16.py         # VGG16 implementation
│
├── Self Supervised Learning SimCLR/
│   ├── dataset_restructuring.py           # Dataset preparation for SimCLR
│   ├── resnet18_fine_tuning_simclr_weights.py      # Fine-tuning with SimCLR
│   └── simclr_unlabeled_weights_training.py        # SimCLR pretraining
│
├── Graphs & Evaluation Metrics/
│   ├── Basic CNN Implementation/
│   │   ├── Patience 5/
│   │   │   ├── Accuracy & Loss Curves.png
│   │   │   ├── Confusion Matrix.png
│   │   │   └── Evaluation Metrics.png
│   │   └── Patience 10/
│   │       ├── Accuracy & Loss Curves.png
│   │       ├── Confusion Matrix.png
│   │       └── Evaluation Metrics.png
│   │
│   ├── Self Supervised Learning SimCLR/
│   │   ├── Contrastive Loss Curves.png
│   │   ├── Feature Embeddings.png
│   │   └── Training Progress/
│   │       ├── Epoch_1.png
│   │       └── Final_Epoch.png
│   │
│   ├── SimCLR ResNet18/
│   │   ├── Accuracy & Loss Curves.png
│   │   ├── Confusion Matrix.png
│   │   ├── Evaluation Metrics.png
│   │   ├── ResNet18 Training on Weights End.png
│   │   ├── ResNet18 Training on Weights Start.png
│   │   └── Sample Pred 1.png
│   │
│   └── Transfer Learning Resnet50/
│       ├── Accuracy & Loss Curves.png
│       ├── Confusion Matrix.png
│       └── Evaluation Metrics.png
│
└── v1 Training Codes/                     # Contains source codes for previous implementations
    ├── old_cnn_implementation.py
    ├── previous_data_processing.py
    └── legacy_training_scripts.py
```

## Methodology

### 1. Basic CNN Implementation
A custom Convolutional Neural Network built from scratch to serve as a baseline model.

#### Architecture
- **Input**: 3x224x224 RGB image
- **Convolutional Layers**:
  - Conv2D(32 filters, 3x3 kernel, padding=1) → BatchNorm → ReLU → MaxPool(2x2)
  - Conv2D(64 filters, 3x3 kernel, padding=1) → BatchNorm → ReLU → MaxPool(2x2)
  - Conv2D(128 filters, 3x3 kernel, padding=1) → BatchNorm → ReLU → MaxPool(2x2)
- **Fully Connected Layers**:
  - Flatten
  - Dense(256) → ReLU → Dropout(0.5)
  - Dense(1) → Sigmoid

#### Implementation Steps
1. **Data Preprocessing & Augmentation**:
   - Image normalization by scaling pixel values to the range [0,1]
   - Data augmentation: random rotations, translations, zooming, and flipping
   - Images resized to 224×224 pixels

2. **Model Compilation**:
   - Optimizer: Adam with default hyperparameters
   - Loss Function: Binary cross-entropy
   - Evaluation Metrics: Accuracy, AUC, Precision, Recall

3. **Training Strategy**:
   - EarlyStopping with patience of 5 epochs
   - ReduceLROnPlateau to reduce learning rate if validation loss stagnates
   - Maximum 100 epochs

### 2. Transfer Learning with ResNet50
Utilizing a pre-trained ResNet50 model to leverage features learned from ImageNet and adapt them to our classification task.

#### Architecture
- **Input**: 3x224x224 RGB image
- **Base Model**: ResNet50 pretrained on ImageNet (excluding top layers)
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense(128) → ReLU
  - Dense(1) → Sigmoid

#### Implementation Steps
1. **Model Setup**:
   - Load ResNet50 pretrained on ImageNet, excluding top layers
   - Initially freeze all base model layers
   - Add custom classification head

2. **Training Process**:
   - First phase: Train only custom layers with learning rate 0.001
   - Second phase: Unfreeze all layers and fine-tune with reduced learning rate (0.0001)
   - Early stopping with patience of 15 epochs
   - Maximum 100 epochs per phase

3. **Additional Models**:
   - Similar transfer learning approach also applied to VGG16 and MobileNet for comparison

### 3. Self-Supervised Learning (SimCLR + ResNet18)
A state-of-the-art self-supervised representation learning framework that eliminates the need for extensive manual labeling.

#### Architecture
- **Encoder**: ResNet18 backbone (outputs 512-d feature vector)
- **Projection Head**: MLP(512 → 256 → 128)
- **For Classification**: Linear(512 → 1) → Sigmoid

#### Implementation Steps

1. **Pretraining Phase**:
   - Generate two augmented views per input image using transformations
   - Pass both views through a shared ResNet18 encoder
   - Project encoder outputs to 128-dimensional latent space
   - Compute contrastive loss (NT-Xent) to learn representations

2. **Fine-Tuning Phase**:
   - Discard projection head
   - Add binary classification head
   - Train on labeled data

3. **Training Configuration**:
   - Adam optimizer with learning rate 0.001
   - Up to 100 epochs with early stopping
   - Batch size of 96

## Experimental Setup

All three models were trained with the following configurations:

- **Data Split**: 70% training, 15% validation, 15% testing
- **Data Augmentation**: Random rotation, width/height shift, zoom, horizontal flip
- **Image Normalization**: Pixel values divided by 255
- **Batch Size**: 32 (CNN and ResNet50), 96 (SimCLR)
- **Early Stopping**: Based on validation loss with patience
- **Loss Function**: Binary cross-entropy (supervised models), NT-Xent (SimCLR)
- **Optimizer**: Adam

## Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Basic CNN | 0.91 | 0.90 | 0.89 | 0.90 | 0.97 |
| ResNet50 (Transfer Learning) | 0.98 | 0.97 | 0.97 | 0.98 | 0.99 |
| SimCLR + ResNet18 | 0.85 | 0.84 | 0.82 | 0.83 | 0.95 |

### Key Findings

1. **Self-Supervised Learning (SimCLR)**:
   - Achieved the best overall performance
   - Particularly effective with limited labeled data
   - Created semantically meaningful representations without labels
   - Required more computational resources

2. **Transfer Learning (ResNet50)**:
   - Performed well with significantly less training time than SimCLR
   - Showed accelerated convergence (reached high accuracy within 6 epochs)
   - Required domain-specific fine-tuning

3. **Basic CNN**:
   - Served as a good baseline but showed limited capacity
   - More sensitive to hyperparameter changes
   - Struggled with subtle texture variations between classes

## Limitations

1. **Self-Supervised Learning**:
   - High computational cost requiring large batch sizes
   - Performance dependent on batch size and number of negative examples

2. **Basic CNN**:
   - Under-parameterized for complex tasks
   - Prone to overfitting despite regularization

3. **Transfer Learning**:
   - Required significant domain-specific fine-tuning
   - Larger model size and latency concerns for deployment

## Future Work

1. **Hybrid Self-Supervised + Transfer Learning**:
   - Combine approaches to bridge domain gaps while retaining rich features
   - Compare SimCLR with alternatives like BYOL and SwAV

2. **Architecture Exploration & Compression**:
   - Evaluate lightweight backbones like EfficientNet Lite and MobileNetV3
   - Apply pruning or knowledge distillation for deployment on resource-constrained devices

3. **Enhanced Data-Level Techniques**:
   - Incorporate stronger augmentation strategies
   - Implement active learning to reduce labeling costs

4. **Robustness & Interpretability Studies**:
   - Conduct adversarial-noise and occlusion tests
   - Leverage explainability tools like Grad CAM and LIME
  
## Dependencies

- TensorFlow 2.x
- PyTorch
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- OpenCV

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection

# Install dependencies
pip install -r requirements.txt

# Run the model training
python train.py --model [cnn|resnet50|simclr]
```

## Running TensorFlow/PyTorch on GPU Locally

To efficiently train deep learning models, we set up GPU acceleration using the following configurations:

### 1. TensorFlow-GPU Support and Compatibility
- Since TensorFlow-GPU support for Windows 10 was discontinued after version **2.10.0**, we ensured compatibility by using **Python 3.9.11** and **TensorFlow 2.10.0**.
- This allows TensorFlow to leverage GPU acceleration efficiently without encountering compatibility issues.

### 2. CUDA and cuDNN Setup
- **CUDA v11.2** and **cuDNN v8.1** are installed to enable GPU acceleration.
- These libraries provide the necessary interface for running deep learning operations on **NVIDIA GPUs**, significantly improving training speed.

### 3. TensorRT for Optimized Inference
- **TensorRT** is installed to optimize inference, enabling faster execution of deep learning models without compromising accuracy.
- This helps in reducing latency during model deployment.

### 4. PyTorch with CUDA
- **PyTorch** is configured to use CUDA for GPU acceleration.
- This ensures that SimCLR training benefits from hardware acceleration, leading to reduced training time and improved performance.

