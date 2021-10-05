# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 01:53:04 2021

@author: HP
"""

import torch

w = torch.ones(1, requires_grad=True)
print(type(w))
x = torch.tensor(1)
y = torch.tensor(2)

y_hat = w*x
print(w*x)
loss = (y_hat - y).pow(2)
loss.backward()
print(w.grad)
lr = 0.1
with torch.no_grad():
    w -= lr*w.grad
w.grad.zero_() 
  
# repeat line 15 to 23 for updating w 