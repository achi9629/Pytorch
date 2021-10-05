# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 05:02:03 2021

@author: HP
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100,
                                            n_features=1,
                                            noise=20,
                                            random_state=4)
# cast to float Tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(-1,1)

# 1) Model
# Linear model f = wx + b
class LineanRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LineanRegression,self).__init__()
        
        self.lin = nn.Linear(input_size, output_size)
        
    def forward(self,x):
        return self.lin(x)

n_samples, n_features = X.shape
input_size = n_features
output_size = 1
model = LineanRegression(input_size, output_size)

# 2) Loss and optimizer
lr = 0.01
epochs = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# 3) Training loop
for epoch in range(epochs):
    
    # Forward pass and loss
    Y_pred = model(X)
    l = loss(Y_pred, Y)
    
    # Backward pass and update
    l.backward()
    optimizer.step()
    
    # zero grad before new step
    optimizer.zero_grad()
    
    if epoch%100 ==0:
        print(f'epoch: {epoch+1}, loss = {l.item():.4f}')

# Plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy,Y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')
plt.show()
