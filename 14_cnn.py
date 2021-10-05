# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 05:49:36 2021

@author: HP
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import numpy as np

#hyper parameters
epochs = 20
batch_size = 4
lr = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
compose = transform.Compose([transform.ToTensor(), 
                             transform.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])


#CIFAR10
train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                           download = True, 
                                           train=True, 
                                           transform = compose)

test_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                          train=False, 
                                          transform = compose)



# Data loader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
len(train_loader)
examples = iter(train_loader)
example_data, example_targets = examples.next()
print(example_data.shape, example_targets.shape)
print((example_targets))

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(image):
#     image = image/2 + 0.5
#     image = image.numpy()
#     plt.imshow(image.transpose((1,2,0)))
#     plt.show()

# imshow(torchvision.utils.make_grid(example_data))

# Convolution Neural Netowrk 
class convnet(nn.Module):
    def __init__(self):
        super(convnet,self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.act = nn.ReLU()
        self.maxp1 = nn.MaxPool2d(2,2)   
        self.conv2 = nn.Conv2d(6, 16, 5)    
        self.fc1 = nn.Linear(16*5*5, 120)       
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self,x):
        
        out = self.conv1(x)
        out = self.act(out)
        out = self.maxp1(out)
        out = self.conv2(out)
        out = self.act(out)   
        out = self.maxp1(out)
        out = out.view(-1,5*5*16)
        out = self.fc1(out)
        out = self.act(out)        
        out = self.fc2(out)
        out = self.act(out)
        out = self.fc3(out)
        
        return out
        
model = convnet()

# Loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# Train the model
n_steps = len(train_loader)

for epoch in range(epochs):
    
    for i,(x,y) in enumerate(train_loader):
        
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        
        # Forward pass
        y_hat = model(x)
        l = loss(y_hat, y)
        
        # Backward and optimize
        l.backward()       
        optimizer.step()
        optimizer.zero_grad()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_steps}], Loss: {l:.4f}')
            
            
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_sample = 0
    n_correct = 0  
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:        
        y_hat = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(y_hat,1)
        
        n_sample += len(labels)
        n_correct += (predicted == labels).sum()
        
        for i in range(batch_size):
            if predicted[i] == labels[i]:
                n_class_correct[labels[i]]+=1
            n_class_samples[labels[i]]+=1
        
    
    print(f'Accuracy of the network on the 10000 test images: {n_correct/n_sample*100:.4f} %')
    
    for i in range(10):
        print(f'Accuracy of {classes[i]}: {n_class_correct[i]/n_class_samples[i]*100:.4f} %')
