import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

np.random.seed(42)
x = np.random.randn(1, 30) # 30 inputs 
# 2. InitializingParameters with my custom architecture
# Layer 1: 30 -> 15 neurons
W1 = np.random.randn(30, 15) * np.sqrt(2 / 30)
b1 = np.zeros((1, 15))
# Layer 2: 15 -> 8 neurons
W2 = np.random.randn(15, 8) * np.sqrt(2 / 15)
b2 = np.zeros((1, 8))
# Layer 3: 8 -> 4 neurons
W3 = np.random.randn(8, 4) * np.sqrt(2 / 8)
b3 = np.zeros((1, 4))
# Layer 4: 4 -> 1 neuron
W4 = np.random.randn(4, 1) * np.sqrt(2 / 4)
b4 = np.zeros((1, 1))

# Forward Pass:
# First Layer
z1 = x @ W1 + b1
a1 = sigmoid(z1)

# Second Layer
z2 = a1 @ W2 + b2
a2 = sigmoid(z2)

# Third Layer
z3 = a2 @ W3 + b3
a3 = sigmoid(z3)

# Fourth Layer
z4 = a3 @ W4 + b4
a4 = sigmoid(z4)

print("Forward pass completed.")
print(a4)
print("x :", x.shape)
print("a1:", a1.shape)
print("a2:", a2.shape)
print("a3:", a3.shape)
print("a4:", a4.shape) 