# Kolmogorov-Arnold Networks (KAN)

This repository contains an implementation of Kolmogorov-Arnold Networks (KAN) using PyTorch. KANs are a neural network architecture designed to approximate multivariate functions through a combination of linear transformations and spline-based layers. The network is particularly effective for tasks requiring smooth and accurate approximations.
You can see more details in [KAN journal paper](https://arxiv.org/abs/2404.19756)

![image](https://github.com/user-attachments/assets/bdb4bb15-f02c-40fe-8847-cc04b0f07caa)

![image](https://github.com/user-attachments/assets/72c4e49b-4d20-4627-b986-08cb23c2844a)


## Features

- **Spline-based Layers**: Each layer uses B-splines for function approximation, enabling smooth and continuous representations.
- **Customizable Architecture**: Easily customize the number of layers, grid size, spline order, and activation functions.
- **Simple Interface**: Define, train, and evaluate models with standard PyTorch modules and utilities.

## Usage
The implementation consists of two primary classes: Layer and KAN.

### Example Usage

```
import torch
import torch.optim as optim
from kan import KAN

# Define model architecture
model = KAN(layers_hidden=[4, 3, 1], grid_size=5, spline_order=3, sigma=0.1)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Load your dataset (for example, the Iris dataset)
X = torch.tensor([...], dtype=torch.float32)  # Input features
y = torch.tensor([...], dtype=torch.float32)  # Target values

# Training loop
n_epochs = 20
for epoch in range(n_epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item()}")
```

## Understanding the Components
* Layer Class: Implements a single spline-based layer using B-splines.
* KAN Class: Stacks multiple Layer instances to form a network. The architecture is defined by the layers_hidden parameter, which specifies the size of each layer.

## B-Spline Computation
The B-splines are computed using recursive definitions, and the spline coefficients are learned during training. This allows the network to approximate complex functions with smooth transitions.

## Dataset Example
In the provided notebook, the network is trained on the Iris dataset after reducing its dimensionality using PCA. The target variable is encoded, and the network is trained to predict the encoded labels.

## Results
The training process outputs the loss at each epoch. A decreasing loss value typically indicates that the model is learning and improving its predictions.
