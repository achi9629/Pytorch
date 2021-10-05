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
x = torch.tensor([1,2,3,4],dtype = torch.float64)
y = torch.tensor([2,4,6,8],dtype = torch.float64)

w = torch.tensor(0.0,dtype=torch.float,requires_grad=True)

# model output
def forward(x):
    return w*x

loss = nn.MSELoss()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
lr = 0.01
epochs = 50

for epoch in range(epochs):
    # predict = forward pass
    y_pred = forward(x)
    
    # loss
    l = loss(y, y_pred)
    
    # calculate gradients = backward pass
    l.backward()
    
    # update weights
    with torch.no_grad():
        w -= lr*w.grad
    
    # zero the gradients after updating
    w.grad.zero_()
    
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
        

print(f'Prediction after training: f(5) = {forward(5):.3f}')
    