# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 02:02:08 2021

@author: HP
"""
# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights

import torch.nn as nn
import torch


# Compute every step manually

# Linear regression
# f = w * x 
# here : f = 2 * x
x = torch.tensor([[1],[2],[3],[4]],dtype = torch.float32)
y = torch.tensor([[2],[4],[6],[8]],dtype = torch.float32)

n_samples, n_features = x.shape
print(f'#samples: {n_samples}, #features: {n_features}')
# 0) create a test sample
X_test = torch.tensor([5], dtype=torch.float32)

# Here we can use a built-in model from PyTorch
input_size = n_features
output_size = n_features

# 1) Design Model, the model has to implement the forward pass!
model = nn.Linear(input_size, output_size)

# Training
lr = 0.01
epochs = 50

# 2) Construct loss and optimizer
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')


#3) Training loop
for epoch in range(epochs):
    # predict = forward pass
    y_pred = model(x)
    
    # loss
    l = loss(y, y_pred)
    
    # calculate gradients = backward pass
    l.backward()
    
    optimizer.step()
    
    # zero the gradients after updating
    optimizer.zero_grad()  # w.grad.zero_() can aslo be used
    
    if epoch % 10 == 0:
        w,b = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0]:.3f}, loss = {l:.8f}')
        

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')
    