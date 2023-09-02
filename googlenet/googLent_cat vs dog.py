import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
from googlenet.googLenet_practice import MyGoogleNet
import zipfile
import shutil
import random


model = MyGoogleNet()
device = "mps" if torch.backends.mps.is_available() else "cpu"

for params in model.parameters():
    #print(params)
    params.requires_grad = False 

#setting the model parameters to fix the data
model.fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024,512)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(512,2)),
    ('output', nn.LogSoftmax(dim = 1))
    ]))
#print(model)

#dataloader function

def load_data(data_folder, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ])
    
    data = torchvision.datasets.ImageFolder(root = data_folder, transform = transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    return data_loader 

#unzipping file
"""
os.chdir('/Users/parkseongbeom/pytorch-test/deeplearning_practice/cat vs dog')
file_list = ['train.zip','test1.zip']


for file in file_list:
    zip_ref = zipfile.ZipFile(file, 'r')
    zip_ref.extractall()
    zip_ref.close()
    
# shifting jpg file to a specific folder
category = []
filenames = os.listdir('/Users/parkseongbeom/pytorch-test/deeplearning_practice/cat vs dog/train')
od = '/Users/parkseongbeom/pytorch-test/deeplearning_practice/cat vs dog'
for file in filenames:
    if file != "cat" and file != "dog" : 
        if os.path.isdir(od + file) == False:
            category = file.split('.')[0]
            if category == 'dog':
                shutil.move('/Users/parkseongbeom/pytorch-test/deeplearning_practice/cat vs dog/train/' + file, '/Users/parkseongbeom/pytorch-test/deeplearning_practice/cat vs dog/train/dog/' + file)
            else:
                shutil.move('/Users/parkseongbeom/pytorch-test/deeplearning_practice/cat vs dog/train/' + file, '/Users/parkseongbeom/pytorch-test/deeplearning_practice/cat vs dog/train/cat/' + file)
        else: pass
"""



data_folder = '/Users/parkseongbeom/pytorch-test/deeplearning_practice/cat vs dog/train'
batch_size = 32
num_workers = 0
dataloader = load_data(data_folder, batch_size, num_workers)

"""
for batch_idx, (samples, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx+1}:")
    print("Samples shape:", samples.shape)
    print("Labels shape:", labels.shape)
    print("---")

"""
#visualization

random_batch = random.choice(list(dataloader))

samples, labels = random_batch

# Generate random indices for images in the batch
num_images = 5  # Number of images to visualize
random_indices = random.sample(range(samples.shape[0]), num_images)

# Visualize the random images
fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

for i, idx in enumerate(random_indices):
    image = samples[idx].numpy().transpose((1, 2, 0))
    label = labels[idx].item()

    axes[i].imshow(image)
    axes[i].set_title(f"Label: {label}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()


# model part
model = model.to(device) #shifting model to gpu
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, amsgrad=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)

epochs = 3
itr = 1
p_itr = 200
model.train()
total_loss = 0
loss_list = []
acc_list = []
for epoch in range(epochs):
    for samples, labels in dataloader:
        samples, labels = samples.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(samples)
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        scheduler.step() 
        
        if itr%p_itr == 0:
            pred = torch.argmax(output, dim=1)
            correct = pred.eq(labels)
            acc = torch.mean(correct.float())
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, acc))
            loss_list.append(total_loss/p_itr)
            acc_list.append(acc)
            total_loss = 0
        
            
        itr += 1

plt.plot(loss_list, label='loss')
plt.plot(acc_list, label='accuracy')
plt.legend()
plt.title('training loss and accuracy')
plt.show()