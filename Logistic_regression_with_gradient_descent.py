import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)

df['target'] = cancer.target

x = df.drop('target', axis=1)
X = np.array(x)
a = np.array(df['target'])
y = np.array(a) 
m, n = X.shape
mu = np.mean(X, axis =0)
sigma = np.std(X, axis=0)
X = (X - mu) / sigma
#applying vectorization for the weights  
w = np.zeros(n)
b = 0
alpha = 0.01
iters = 20000

cost_history = []
for i in range(iters):
    z= np.dot(X, w) + b
    s= 1/(1 + np.exp(-z)) 
    s = np.clip(s, 1e-15, 1 - 1e-15) # Prevents log(0) errors
    error = s - y
    cost = -1/m * np.sum(y *np.log(s) + (1 -y) * np.log(1 - s))
    dj_dw = 1/m * np.dot(X.T, error)
    dj_db = 1/m * np.sum(error)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db
   
    if i % 100== 0:
        cost_history.append(cost)
        print(f'Step{i}: Cost= {cost:.2f} | w = {w} | b = {b:.2f}')


plt.figure(figsize=(8, 5))
plt.plot(range(0, iters, 100), cost_history, marker='+', color='blue')
plt.xlabel('Iters')
plt.ylabel('Cost')
plt.title('Cost Function Over Time')
plt.show()