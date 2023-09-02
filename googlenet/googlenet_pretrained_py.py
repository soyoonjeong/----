# %%
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
import urllib
from googlenet_pytorch import GoogLeNet 

%matplotlib inline

# %%

model = GoogLeNet()
map_location=torch.device('cpu')
model.load_state_dict(torch.load("./model/googlenet_pretrained.pt"), strict=False)
model.eval()

# %%
input_image = Image.open("./dataset/cat.jpg")
preprocess = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])
input_tensor = preprocess(input_image)
input_tensor = input_tensor.reshape(1, 3, 128, 128)
output = model(input_tensor)
pred = torch.argmax(output, dim=1)
pred = [p.item() for p in pred]
if pred[0] == 0:
    print("cat")
else:
    print("dog")

# %%
input_image = Image.open("./dataset/dog.jpg")
preprocess = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])
input_tensor = preprocess(input_image)
input_tensor = input_tensor.reshape(1, 3, 128, 128)
output = model(input_tensor)
pred = torch.argmax(output, dim=1)
pred = [p.item() for p in pred]
if pred[0] == 0:
    print("cat")
else:
    print("dog")

# %%
         
def plot(image):
    n_features = image.size()[0]
    fig = plt.figure() 
    cols = 16
    rows = n_features//cols
    i = 1

    for idx in range(n_features):
        img = image[idx].detach().numpy()
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([]), ax.set_yticks([])
        i += 1
    
    plt.show()


