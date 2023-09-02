import torch
from googlenet.googLenet_practice import MyGoogleNet
import torch.nn as nn
import matplotlib.pyplot as plt
import urllib
from PIL import Image
from torchvision import transforms
from googlenet.googLenet_practice import random_plot
from googlenet.googLenet_practice import plot

if __name__ == '__main__':
    # Temporary define data and target
    
        
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)


    input_image = Image.open(filename)
    preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    batch_size = 1024 # create a mini-batch as expected by the model

    #x = torch.randn((batch_size, 3, 224, 224))
    #y = torch.randint(0,1000, (batch_size,))
    # x= input_tensor
    # y= torch.randint(0,1000, (batch_size,))
    # y_ = torch.randint(0,1000, (128,))
    # num_classes = 1000
    
    model = models.googlenet
    model.load_state_dict(torch.load("./model/googlenet_dogcat.pt", map_location=torch.device('cpu')))

    model.eval()
    # print(m)
    # we have x,o1,o2 = m(x)
    # m(x)[0] means x; m(x)[1] means o1; m(x)[2] means o2
    # o1 and o2 are output from auxclassifier
    # print(m(x)[0].shape)
    # m.eval()
    # print(m.training)

    # # Notice here! When you going to train your network
    # # Put these loss value into train step of your model
    # m.train()
    # # %%
    # loss = nn.CrossEntropyLoss()
    # loss1 = nn.CrossEntropyLoss()
    # loss2 = nn.CrossEntropyLoss()
    # discount = 0.3

    # o,o1,o2 = m(x)

    # total_loss = loss(o,y) + discount*(loss1(o1,y_) + loss2(o2,y_))
    # print(total_loss)

    # # And while inferencing the model, set the model into
    # # model.eval() mode
    # m.eval()