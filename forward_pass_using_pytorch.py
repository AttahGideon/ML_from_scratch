import torch
import torch.nn as nn
import numpy as np

np.random.seed(42)
x = np.random.randn(1, 30) # 30 inputs 
x = torch.tensor(x, dtype=torch.float32)

# Define the neural network architecture using PyTorch
model = nn.Sequential(
    nn.Linear(30, 15), # Layer 1: 30 -> 15 neurons 
    nn.Sigmoid(),
    nn.Linear(15, 8),  # Layer 2: 15 -> 8 neurons 
    nn.Sigmoid(),
    nn.Linear(8, 4),   # Layer 3: 8 -> 4 neurons
    nn.Sigmoid(),
    nn.Linear(4, 1),   # Layer 4: 4 -> 1 neuron
)

train = model(x)
print(x.shape)
print(train)
print(train.shape)