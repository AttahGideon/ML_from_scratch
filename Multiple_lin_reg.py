import numpy as np 
import matplotlib.pyplot as plt
x = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y = np.array([460, 232, 178])
m, n= x.shape

# Scale features so gradient descent stays numerically stable.
mu = np.mean(x, axis=0)
sigma = np.std(x, axis=0)
x = (x - mu) / sigma

w = np.zeros(n)
b =  0
alpha = 0.01
iters = 10000
cost_history =[]

for i in range(iters):
    predictions = np.dot(x, w) + b
    error = predictions - y
    squared_error = error**2
    cost = 1/(2*m) * np.sum(squared_error)
    dj_dw = 1/m * np.dot(x.T, error)
    dj_db = 1/m * np.sum(error)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db

    if i % 1000 == 0:
        cost_history.append(cost)
        print(f'Step{i}: Cost= {cost:.2f} | w = {w} | b = {b:.2f}')

print('\n...Final Model...')
print(f'Final equation: = {w} * x + {b:.2f}')

plt.figure(figsize=(8, 5))
plt.plot(range(0, iters, 1000), cost_history, marker='+', color='blue')
plt.xlabel('Iters')
plt.ylabel('Cost')
plt.title('Cost Function Over Time')
plt.show()
