# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 12:26:01 2021

@author: HP
"""

import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# 0) Prepare data
bc = datasets.load_breast_cancer()
data, target = bc.data, bc.target

n_samples, n_features = data.shape
x_train,x_test,y_train,y_test = train_test_split(data,target,random_state=0,
                                                 test_size=0.2)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

# scale
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

# cast to float Tensor
X_train = torch.from_numpy(x_train.astype(np.float32))
X_test = torch.from_numpy(x_test.astype(np.float32))
Y_train = torch.from_numpy(y_train.astype(np.float32))
Y_test = torch.from_numpy(y_test.astype(np.float32))
Y_train = Y_train.view(-1,1)
Y_test = Y_test.view(-1,1)

# 1) Model
# Linear model f = wx + b , sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression,self).__init__()
        
        self.lin = nn.Linear(input_size, output_size)
        self.act = nn.Sigmoid()
        
    def forward(self,x):
        x = self.lin(x)
        x = self.act(x)
        return x

input_size = n_features
output_size = 1
model = LogisticRegression(input_size, output_size)

# 2) Loss and optimizer
lr = 0.01
epochs = 100

loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# 3) Training loop
for epoch in range(epochs):
     
     # Forward pass and loss
     Y_pred = model(X_train)   
     l = loss(Y_pred, Y_train)
     
     # Backward pass and update
     l.backward()  
     optimizer.step()
    
    # zero grad before new step
     optimizer.zero_grad()
     
     if (epoch+1)%10==0:
         print(f'epoch: {epoch+1}, loss = {l.item():.4f}')
         
predicted = model(X_test)
predicted = predicted.round()
acc = (predicted == Y_test).sum()/X_test.shape[0]
print(f' accuracy : {acc.item()}')
