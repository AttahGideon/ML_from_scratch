import torch
import torch.nn as nn
import torch.optim as optim

#Distance in Km
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype = torch.float32) 

times = torch.tensor([[6.96], [12.11], [16.71], [22.21]], dtype = torch.float32)

model = nn.Sequential(nn.Linear(1, 1))
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr =0.01)

#Training loop
for epoch in range(500):
    #reset the optimizer
    optimizer.zero_grad()
    #make predictions
    outputs = model(distances)
    #Calculate loss
    loss = loss_function(outputs, times)
    #calculate adjustments
    loss.backward()
    #update weights
    optimizer.step()


with torch.no_grad():
    test_run =torch.tensor([[3.50]], dtype = torch.float32)
    predict_time = model(test_run)

print(f'Predicted time for 3.5 Km is: {predict_time.item():.1f} minutes')