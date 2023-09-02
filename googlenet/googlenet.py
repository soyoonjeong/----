import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import urllib
from PIL import Image 
from torchvision import transforms 
import matplotlib.pyplot as plt
import random
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def random_plot(image):
    n_features = image.size()[0]
    size = image.size()[1]
    images_per_row = 16
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    fig = plt.figure() # rows*cols 행렬의 i번째 subplot 생성
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



def plot_image(image):
    # Normalize the image to [0, 1] range
    selected_channels = image[..., :3]  # Select the first three channels

    # Normalize the image to [0, 1] range
    selected_channels = selected_channels.astype(np.float32) / 255.0

    # Plot the image
    plt.imshow(selected_channels)
    plt.axis('off')
    plt.show()

def imshow(torch_img):
    # Convert the tensor to a NumPy array
    np_array = torch_img.detach().numpy()

    # Get the dimensions of the tensor
    dimensions = torch_img.size()

    # Reshape the array for visualization
    if len(dimensions) == 4:
        reshaped_array = np_array.transpose(0, 2, 3, 1)
        reshaped_array = reshaped_array.reshape(dimensions[0] * dimensions[2], dimensions[3], dimensions[1])

        # Display the images
        fig, axes = plt.subplots(nrows=dimensions[0], ncols=dimensions[2], figsize=(12, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(reshaped_array[i], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    elif len(dimensions) == 3:
        reshaped_array = np_array.transpose(1, 2, 0)
        plot_image(reshaped_array) 

    

def show(*imgs):
    '''
     input imgs can be single or multiple tensor(s), this function uses matplotlib to visualize.
     Single input example:
     show(x) gives the visualization of x, where x should be a torch.Tensor
        if x is a 4D tensor (like image batch with the size of b(atch)*c(hannel)*h(eight)*w(eight), this function splits x in batch dimension, showing b subplots in total, where each subplot displays first 3 channels (3*h*w) at most. 
        if x is a 3D tensor, this function shows first 3 channels at most (in RGB format)
        if x is a 2D tensor, it will be shown as grayscale map
     
     Multiple input example:      
     show(x,y,z) produces three windows, displaying x, y, z respectively, where x,y,z can be in any form described above.
    '''
    img_idx = 0
    for img in imgs:
        img_idx +=1
        plt.figure(img_idx)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()

            if img.dim()==4: # 4D tensor
                bz = img.shape[0]
                c = img.shape[1]
                if bz==1 and c==1:  # single grayscale image
                    img=img.squeeze()
                elif bz==1 and c==3: # single RGB image
                    img=img.squeeze()
                    img=img.permute(1,2,0)
                elif bz==1 and c > 3: # multiple feature maps
                    img = img[:,0:3,:,:]
                    img = img.permute(0, 2, 3, 1)[:]
                    print('warning: more than 3 channels! only channels 0,1,2 are preserved!')
                elif bz > 1 and c == 1:  # multiple grayscale images
                    img=img.squeeze()
                elif bz > 1 and c == 3:  # multiple RGB images
                    img = img.permute(0, 2, 3, 1)
                elif bz > 1 and c > 3:  # multiple feature maps
                    img = img[:,0:3,:,:]
                    img = img.permute(0, 2, 3, 1)[:]
                    print('warning: more than 3 channels! only channels 0,1,2 are preserved!')
                else:
                    raise Exception("unsupported type!  " + str(img.size()))
            elif img.dim()==3: # 3D tensor
                bz = 1
                c = img.shape[0]
                if c == 1:  # grayscale
                    img=img.squeeze()
                elif c == 3:  # RGB
                    img = img.permute(1, 2, 0)
                else:
                    raise Exception("unsupported type!  " + str(img.size()))
            elif img.dim()==2:
                pass
            else:
                raise Exception("unsupported type!  "+str(img.size()))


            img = img.numpy()  # convert to numpy
            img = img.squeeze()
            if bz ==1:
                plt.imshow(img, cmap='gray')
                # plt.colorbar()
                # plt.show()
            else:
                for idx in range(0,bz):
                    plt.subplot(int(bz**0.5),int(np.ceil(bz/int(bz**0.5))),int(idx+1))
                    plt.imshow(img[idx], cmap='gray')

        else:
            raise Exception("unsupported type:  "+str(type(img)))




class ConvBlock(nn.Module):
    def __init__(self, in_fts, out_fts, k, s, p):
        super(ConvBlock, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(k, k), stride=(s, s), padding=(p, p)),
            nn.ReLU()
        )

    def forward(self, input_img):
        x = self.convolution(input_img)
        random_plot(x)
        return x


class ReduceConvBlock(nn.Module):
    def __init__(self, in_fts, out_fts_1, out_fts_2, k, p):
        super(ReduceConvBlock, self).__init__()
        self.redConv = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts_1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_fts_1, out_channels=out_fts_2, kernel_size=(k, k), stride=(1, 1), padding=(p, p)),
            nn.ReLU()
        )

    def forward(self, input_img):
        x = self.redConv(input_img)

        return x


class AuxClassifier(nn.Module):
    def __init__(self, in_fts, num_classes):
        super(AuxClassifier, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3))
        self.conv = nn.Conv2d(in_channels=in_fts, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 4 * 128, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.avgpool(input_img)
        x = self.conv(x)
        x = self.relu(x)
        x = x.reshape(N, -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


class InceptionModule(nn.Module):
    def __init__(self, curr_in_fts, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pool_proj):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvBlock(curr_in_fts, f_1x1, 1, 1, 0)#conv 1x1+1 -> 64(output)
        self.conv2 = ReduceConvBlock(curr_in_fts, f_3x3_r, f_3x3, 3, 1) # 1x1 -> 96(output) -> 3x3 ->128(output)
        self.conv3 = ReduceConvBlock(curr_in_fts, f_5x5_r, f_5x5, 5, 2) # 1x1 -> 16(output) -> 5x5 -> 32(outputs)

        self.pool_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=curr_in_fts, out_channels=f_pool_proj, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()
        ) #

    def forward(self, input_img):
        out1 = self.conv1(input_img)
        out2 = self.conv2(input_img)
        out3 = self.conv3(input_img)
        out4 = self.pool_proj(input_img)

        x = torch.cat([out1, out2, out3, out4], dim=1)

        return x


class MyGoogleNet(nn.Module):
    def __init__(self, in_fts=3, num_class=1000):
        super(MyGoogleNet, self).__init__()
        self.conv1 = ConvBlock(in_fts, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Sequential(
            ConvBlock(64, 64, 1, 1, 0), #### 입력 채널의 크기 == 출력 채녈의 크기 -> 해당 층의 역할??
            ConvBlock(64, 192, 3, 1, 1) #
        )

        self.inception_3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.inception_4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.inception_5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.aux_classifier1 = AuxClassifier(512, num_class)
        self.aux_classifier2 = AuxClassifier(528, num_class)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024 * 7 * 7, num_class)
        )

    def forward(self, input_img):
        N = input_img.shape[0]

        #part1
        x = self.conv1(input_img)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)

        x = self.inception_3a(x)
        
        x = self.inception_3b(x)
        x = self.maxpool1(x)
        x = self.inception_4a(x)
        out1 = self.aux_classifier1(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        out2 = self.aux_classifier2(x)
        x = self.inception_4e(x)
        x = self.maxpool1(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        if self.training == True:
            return [x, out1, out2]
        else:
            return 
        

if __name__ == '__main__':
    # Temporary define data and target
    batch_size = 5
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)


    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    batch_size = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    #x = torch.randn((batch_size, 3, 224, 224))
    #y = torch.randint(0,1000, (batch_size,))
    x= input_tensor
    y= input_tensor
    num_classes = 1000
    #x = torch.randn((batch_size, 3, 224, 224))
    #y = torch.randint(0,1000, (batch_size,))
    #num_classes = 1000

    # Add to graph in tensorboard
    writer = SummaryWriter(log_dir='logs/googlenet')
    m = MyGoogleNet()
    # print(m)
    # we have x,o1,o2 = m(x)
    # m(x)[0] means x; m(x)[1] means o1; m(x)[2] means o2
    # o1 and o2 are output from auxclassifier
    print(m(x)[0].shape)
    m.eval()
    print(m.training)
    writer.add_graph(m, x)
    writer.close()

    # Notice here! When you going to train your network
    # Put these loss value into train step of your model
    m.train()
    loss = nn.CrossEntropyLoss()
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    discount = 0.3

    o,o1,o2 = m(x)

    total_loss = loss(o,y) + discount*(loss1(o1,y) + loss2(o2,y))
    print(total_loss)

    # And while inferencing the model, set the model into
    # model.eval() mode
    m.eval()
    