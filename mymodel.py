import torch
import torch.nn as nn
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:

        self.ln_in = 300
        self.ln_out = 2

        self.conv1 = nn.Conv2d(1, 5, (1, 10), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(5)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(5, 5, (1, 10), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(5, 5, (1, 11), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(5)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(5, 5, (1, 11), stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(5)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(5, 5, (3, 8), stride=(2, 2))
        self.bn5 = nn.BatchNorm2d(5)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(5, 5, (4, 7), stride=(2, 1))
        self.bn6 = nn.BatchNorm2d(5)
        self.relu6 = nn.ReLU(inplace=True)

        #之后加上lstm+maxpooling+dropout+fc+relu
        self.lstm = RNN()
        self.maxpooling = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(490,self.ln_out)
        self.relu7 =  nn.ReLU(inplace=True)



    def forward(self, x,h_n,h_c):
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

        #or lstm,x's dimension is 3
        #so squeeze dim 1
        # b,c,h,w = x.shape
        x = x.reshape(x.shape[0],98,10)

        self.lstm.to(device=x.device)
        x,h_n,h_c = self.lstm(x,h_n,h_c)

        #reshape x from 3 dimension to 4 demension
        b1,c1,hw = x.shape
        x = x.reshape(b1,5,4,98)
        x = self.maxpooling(x)
        x = self.drop(x)
        
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = self.relu7(x)
        return x,h_n,h_c

class RNN(nn.Module):
    def __init__(self,input_size=10):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=20,
            num_layers=1,
            batch_first=False
        )
        # num_layers = 1,
        # batch_first = True


    def forward(self, x,h_n,h_c):
        # input = torch.randn((18, 5, 196),device=x.device)
        # h0 = torch.randn((1, 5, 20),device=x.device,dtype=torch.float32)
        # c0 = torch.randn((1, 5, 20),device=x.device,dtype=torch.float32)
        # r_out, (h_n, h_c) = self.rnn(input, (h0, c0))
        r_out, (h_n, h_c) = self.rnn(x, (h_n, h_c))  # None 表示 hidden state 会用全0的 state
        return r_out,h_n,h_c