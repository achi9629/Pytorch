# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:02:32 2021

@author: HP
"""

import torch
import numpy as np

#1 garbage values of size (2,3)
a=torch.empty((2,3), dtype=torch.int32, device = 'cpu')
print('1',a)

#2 random values of size (2,3)
b = torch.rand(2,3, dtype = torch.float32, device = 'cpu')
print('2',b)

#3 ones and zeroes of size (2,3)
c = torch.ones(2,3,dtype = torch.float32, device = 'cpu')
d = torch.zeros(2,3,dtype = torch.float32, device = 'cpu')
print('3',c,d)

#4 numpy to tensor
print('4',torch.tensor(np.array([1,2,3,4]), dtype = torch.int32, device = 'cpu'))

#5 add 2 tensors
x = torch.rand(2,2)
y = torch.rand(2,2)
print('5.1',x,y)
z = x + y
print('5.2',z)
z = torch.add(x,y)
print('5.3',z)
y.add_(x)                 #y = y + x
print('5.4',y)

#6 subtract 2 tensors
x = torch.rand(2,2)
y = torch.rand(2,2)
print('6.1',x,y)
z = x - y
print('6.2',z)
z = torch.sub(x,y)
print('6.3',z)
y.sub_(x)                #y = y - x
print('6.4',y)

#7 multiply 2 tensors
x = torch.rand(2,2)
y = torch.rand(2,2)
print('7.1',x,y)
z = x * y
print('7.2',z)
z = torch.mul(x,y)
print('7.3',z)
y.mul_(x)              # y = y * x
print('7.4',y)

#8 divide 2 tensord
x = torch.rand(2,2)
y = torch.rand(2,2)
print('8.1',x,y)
z = x / y
print('8.2',z)
z = torch.div(x,y)
print('8.3',z)
y.div_(x)             # y = y / x
print('8.4',1/y)

#9 slicing
x = torch.rand(5,5)
print('9.1',x)
print('9.2',x[2:4,1:3])
print('9.3',x[1,1].item()) # item() is used only when one value is in slice


#10 reshape
x = torch.rand(5,3)
print('10.1',x)
y = x.view(15)
print('10.2',y)
y = x.view(3,5)
print('10.2',y)
y = x.view(3,-1)           # Using -1 pytorch will automatically determine other diminsion 
print('10.4',y)
y = x.view(-1,3)
print('10.5',y,y.size())

#11 convert numpy to tensor
a = np.ones(5)
print('11.1',a)
b = torch.from_numpy(a)
print('11.2',b)

a+=1                         # On CPU both a and b share the same memory location
print('11.3',a)              # a and b will have same value
print('11.4',b)

b+=1
print('11.5',a)             # a and b will have same value
print('11.6',b)             # a and b will have same value

#12 convert tensor to numpy
a = torch.ones(5)
print('12.1',a)
b = a.numpy()
print('12.2',b)

a+=1
print('12.3',a)             # a and b will have same value
print('12.4',b)             # a and b will have same value

b+=1
print('12.5',a)             # a and b will have same value
print('12.6',b)             # a and b will have same value