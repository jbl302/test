# Neural Network Implementations and Progression

## Overview
This repository contains implementations of neural networks for breast cancer detection using PyTorch. The project progresses from a basic neural network implemented manually using tensors to a more modular approach using `torch.nn` and `torch.optim`.

## Implementations

### 1. **Manual Implementation (Without `torch.nn`)**
- Implemented forward and backward propagation manually using PyTorch tensors.
- Used basic operations to compute gradients and update weights.
- Loss function: Binary Cross-Entropy (BCE)
- Optimizer: Gradient Descent (manually implemented)
- Dataset: Breast cancer dataset from `sklearn.datasets`.

### 2. **Using `torch.nn` (Modular Approach)**
- Defined a neural network using `torch.nn.Module`.
- Utilized `torch.nn.Linear` layers and activation functions like ReLU and Sigmoid.
- Loss function: `torch.nn.BCELoss()`
- Optimizer: `torch.optim.Adam` for efficient weight updates.
- Improved code readability and maintainability.

## Dataset
- Features: Various attributes related to breast cancer diagnosis.
- Target: Binary classification (Malignant = 1, Benign = 0).
- Preprocessing: Standardized features using `StandardScaler()` from `sklearn.preprocessing`.

## Training Pipeline
1. Load and preprocess dataset.
2. Split into training and testing sets.
3. Define the neural network architecture.
4. Train using backpropagation and optimization techniques.
5. Evaluate using accuracy, precision, and recall.

