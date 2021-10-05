# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 00:18:26 2021

@author: HP
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt

#hyper parameters
input_size = 784 #28*28
output_size = 10
hidden_size = 100
epochs = 10
batch_size = 100
lr = 0.001


compose = transform.Compose([transform.ToTensor(),transform.Normalize((.1307,), (.3081))])
#MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           download = True, 
                                           train=True, 
                                           transform = compose)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform = compose)



# Data loader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

examples = iter(train_loader)
example_data, example_targets = examples.next()
print(example_data.shape, example_targets.shape)
print(len(example_targets))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size)

# Loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr)

# Train the model
n_steps = len(train_loader)

for epoch in range(epochs):
    
    for i,(x,y) in enumerate(train_loader):
        
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        x = x.view(-1,784)
        
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
    for images, labels in test_loader:        
        images = images.view(-1,28*28)
        y_hat = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(y_hat,1)
        
        n_sample += len(labels)
        n_correct += (predicted == labels).sum()
        
    
    print(f'Accuracy of the network on the 10000 test images: {n_correct/n_sample*100:.4f} %')


torch.save(model.state_dict(),'mnist_ffn.pth')
model.state_dict()
