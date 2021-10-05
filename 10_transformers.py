# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:20:31 2021

@author: HP
"""
'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html
On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale
On Tensors
----------
LinearTransformation, Normalize, RandomErasing
Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage
Generic
-------
Use Lambda 
Custom
------
Write own class
Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''


import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision

class WineDataset(Dataset):
    def __init__(self, transform = None):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('./data/wine.csv',dtype = np.float32, delimiter=',',skiprows=1)
        self.num_samples = xy.shape[0]
        
        # here the first column is the class label, the rest are the features
        self.x_data = xy[:,1:]   # size [n_samples, n_features]
        self.y_data = xy[:,[0]]  # size [n_samples, 1]
        
        self.transform = transform
        # support indexing such that dataset[i] can be used to get i-th sample
        
    def __getitem__(self, index):
            data  = self.x_data[index], self.y_data[index]
            if self.transform:
                data = self.transform(data)
            return data
        
        # we can call len(dataset) to return the size
    def __len__(self):
            return self.num_samples

# Custom Transforms
# implement __call__(self, sample)
class ToTensor:
    
    # Convert ndarrays to Tensors
    def __call__(self,sample):
        inputs, outputs = sample
        return torch.from_numpy(inputs), torch.from_numpy(outputs)
    
class MulTransform:
    
     # multiply inputs with a given factor
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        features, target = sample;
        features *= self.factor
        return features, target


print('Without Transform')    
dataset = WineDataset(transform=None)
first_data = dataset[0]
features, target = first_data
print(type(features), type(target))
print(features, target)

print('\nWith Tensor Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, target = first_data
print(type(features), type(target))
print(features, target)

print('\nWith Tensor and Multiplication Transform')
compose = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform = compose)
first_data = dataset[0]
features, target = first_data
print(type(features), type(target))
print(features, target)
