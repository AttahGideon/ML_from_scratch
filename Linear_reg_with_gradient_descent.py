import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\hp\Linear Regression & Correlation\Salary_dataset.csv")

x = df['YearsExperience'].values
y = df['Salary'].values
m = len(x)

#base weights and bias
w = 0
b = 0

# Base Hyperparameters
alpha = 0.00005
iterations = 1500000

# Main training loop
for i in range(iterations):
    predictions = w * x + b
    error = predictions - y
    squared_error = error**2
    cost = 1/(2*m) * np.sum(squared_error)

    dj_dw = 1/m * np.sum(error * x)
    dj_db = 1/m * np.sum(error)

    w = w - alpha * dj_dw
    b = b - alpha * dj_db

    if i % 1000 == 0:
        print(f'Step{i}: Cost= {cost:.2f} | w = {w:.2f} | b = {b:.2f}')

print('\n...Final Model...')
print(f'Final equation: Salary = {w:.2f} * YearsExperience + {b:.2f}')
