# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 23:14:51 2021

@author: HP
"""

# output = w*x + b
# output = activation_function(output)
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-1.0, 1.0, 2.0, 3.0])

#softmax

output = torch.softmax(x, 0)
print(output)

output = nn.Softmax(0)(x)
print(output)



#sigmoid

output = torch.sigmoid(x)
print(output)

output = nn.Sigmoid()(x)
print(output)



#tanh

output = torch.tanh(x)
print(output)

output = nn.Tanh()(x)
print(output)



#Relu

output = torch.relu(x)
print(output)

output = nn.ReLU()(x)
print(output)




#Leaky Relu

output = F.leaky_relu(x,0.11)
print(output)

output = nn.LeakyReLU(0.11)(x)
print(output)




#nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model.
#torch.relu on the other side is just the functional API call to the relu function,
#so that you can add it e.g. in your forward method yourself.

# option 1 (create nn modules)

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNet, self).__init__()
        
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        y_pred = self.sigmoid(out)
        
        return y_pred
    
# option 2 (use activation functions directly in forward pass)
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNet, self).__init__()
        
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
        
    def forward(self,x):
        out = torch.ReLU(self.lin1(x))
        out = torch.sigmoid(self.lin2(out))
        
        return out