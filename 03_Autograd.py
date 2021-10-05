# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 01:10:03 2021

@author: HP
"""

import torch

#1 finding gradient of x with scalar at the end
x = torch.rand(3, requires_grad=True) #If autograd should record operations 
print(x)                              #on the returned tensor. Default: False.
                            
y = x + 2
print(y)

z1 = 2*y*y
print(z1)

z2 = z1.mean()
print(z2)

z2.backward()

print(x.grad)

#2 finding gradient of x with vector at the end
x = torch.rand(3, requires_grad=True) #If autograd should record operations 
print(x)                              #on the returned tensor. Default: False.
                            
y = x + 2
print(y)

z1 = 2*y*y
print(z1)

# z2 = z1.mean()
# print(z2)

v = torch.tensor([0.1,1,1],dtype = torch.int32)  # if z1 is not scalar need to 
z1.backward(v)                                  #include a vector ot this size

print(x.grad)           # Dont know why this output is coming


#3 Disable gradient, stop pytorch from creating gradient function
x = torch.rand(5, requires_grad=True)
print(x)

y = x.requires_grad_(False)   # Inplace
print(y)
print(x)

x.requires_grad_(True)        #Inplace
y = x.detach_()
print(y)
print(x)

with torch.no_grad():
    y = x + 1
    print(y)
    print(x)

#4 not add gradient weight = 0
weight = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (3*weight).mean()
    model_output.backward()
    print(weight.grad)                    # weights got sum up after each epoch
    
#5 not add gradient weight = 0
weight = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (3*weight).mean()
    model_output.backward()
    print(weight.grad)
    weight.grad.zero_()                  #need to make weight zero after each epoch
    
