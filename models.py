## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

from fractions import gcd

class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32, act=True):
        super(Linear, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.ln_in = 300
        self.ln_out = 2

        self.conv1 = nn.Conv2d(1, 5, (1,10),stride=(1,2))
        self.bn1 = nn.BatchNorm2d(5)
        self.relu1 = nn.ReLU(inplace=True)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        #self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(5,5,(1,10),stride=(1,3))
        self.bn2 = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU(inplace=True)


        self.conv3 = nn.Conv2d(5,5,(1,11),stride=(1,3))
        self.bn3 = nn.BatchNorm2d(5)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(5,5,(1,11),stride=(1,2))
        self.bn4 = nn.BatchNorm2d(5)
        self.relu4 = nn.ReLU(inplace=True)


        self.conv5 = nn.Conv2d(5,5,(3,8),stride = (2,2))
        self.bn5 = nn.BatchNorm2d(5)
        self.relu5 = nn.ReLU(inplace=True)


        self.conv6 = nn.Conv2d(5,5,(4,7),stride=(2,1))
        self.bn6 = nn.BatchNorm2d(5)
        self.relu6 = nn.ReLU(inplace=True)


        self.bn7 = nn.BatchNorm2d(5)


        
        # self.fc1 = nn.Linear(512*6*6, 1024)
        # self.fc2 = nn.Linear(1024, 136)
        self.fc1 = Linear(self.ln_in,self.ln_out,norm = 'BN')
        


        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)



        x = self.bn7(x)
        x = x.view(-1,300)
        x = self.fc1(x)


      
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = self.drop1(x)
        # x = self.pool2(F.relu(self.conv2(x)))
        # x = self.drop2(x)
        # x = self.pool3(F.relu(self.conv3(x)))
        # x = self.drop3(x)
        # x = self.pool4(F.relu(self.conv4(x)))
        # x = self.drop4(x)
        # x = self.pool5(F.relu(self.conv5(x)))
        # x = self.drop5(x)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = self.drop6(x)
        # x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
