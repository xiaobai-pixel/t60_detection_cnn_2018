import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

import torchvision.transforms

from workspace_utils import active_session
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import valdata_meant60

# 决定使用哪块GPU进行训练
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from mymodel import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim
import datetime
from valdata_meant60 import ValDataset, Val_meanT60


def net_sample_output():
    for i, sample in enumerate(val_loader):

        images = sample['image']
        ddr = sample['ddr']
        t60 = sample['t60']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = torch.tensor(images.clone().detach(), dtype=torch.float32, device=device)

        # forward pass to get net output
        t60_predict = net(images)

        # break after first image is tested
        if i == 0:
            return images, t60_predict, t60


def train_net(n_epochs,train_loader,batch_size):

    start_epoch=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = torch.optim.lr_scheduler.StepLR(optimizer,
                                         step_size=150,
                                         gamma=0.33)

    lr_list = list()
    for epoch in range(start_epoch,n_epochs):
        print("Training:epoch ", epoch)
        lr_list.append(lr.get_last_lr())
        running_loss = 0.0
        # 做个一个batch，将计算出来的loss与之前的loss相加计算总的Loss
        Bias_all_batch = 0.0
        count_num = 0
        Mse_all_batch = 0.0

        #遍历train_loader,其中k是voiceName,data是list，里面存储了多个字典
        #因为它们是属于同一个声音的
        num = 0
        total_num = 0
        #计算做了几个batch，防止最后那部分数据没有做反向传播
        c_batch = 0
        loss = 0
        for k,datas in train_loader.items():
            if num >= batch_size:
                loss = 0
            begin = time.time()
            #遍历同一个声源的内容，并作为一个batch_size送入网络
            input = []
            label = []
            if epoch == 0:
                h_n = torch.randn((1,98, 20),device=device,dtype=torch.float32)
                h_c = torch.randn((1,98, 20),device=device,dtype=torch.float32)
            for data in datas:
                images = data['image']
                ddr = data['ddr']
                meanT60 = data['MeanT60']
                input.append(images)
                label.append(meanT60)
            images = torch.stack(input,dim=0)
            meanT60 = torch.stack(label,dim=0)
            num += images.shape[0]
            total_num += images.shape[0]
            loss = 0


            # 如果使用cuda的话，好像会默认会用cuda:0，所以我用torch.tensor来定义dtype,device
            meanT60 = torch.tensor(meanT60.clone().detach().numpy(), dtype=torch.float32, device=device)
            images = torch.tensor(images.clone().detach().numpy(), dtype=torch.float32, device=device)
            # output_pts = net(images)
            output_pts,h_n,h_c = net(images,h_n,h_c)
            loss =  criterion(output_pts, meanT60)
            #当参与训练的图片数量大于500时就要做反向传播
            # optimizer.zero_grad()
            # # with torch.autograd.set_detect_anomaly(True):
            # #     loss.backward(retain_graph=True)
            #
            # # torch.autograd.set_detect_anomaly(True)
            # loss.backward()
            # optimizer.step()
            if num >= batch_size:
                optimizer.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                loss.backward(retain_graph=True)
                optimizer.step()
                c_batch += 1
                num = 0
            # if (total_num//batch_size <= c_batch) and (total_num//batch_size+1 >= c_batch):
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            #     c_batch += 1
            #     num = 0

            bias =  torch.sum((meanT60 -output_pts))/output_pts.shape[0]/output_pts.shape[1]

        print("epoch {},mse is {},bias is {}".format(epoch,loss.item(),bias.item()))
        lr.step()
        # if epoch%10 == 3:
        #
        #     val = Val_meanT60()
        #     mse_way1, mse_way2, bias_way1, bias_way2 = val(epoch, net, val_loader, lr.get_last_lr(),device)
        #     print("at eval,mean_mse_way1 is {},mean_mse_way2 is {},"
        #           "mean_bias_way1 is {},mean_bias_way2 is {}".format(mse_way1, mse_way2, bias_way1, bias_way2))

        if epoch % 10 == 3:
            model_dir = './Checkpoints/relative_loss_addBias/'
            model_name = 't60_predict_model_%d_meanT60_continue11.pt' % (epoch)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            # after training, save your model parameters in the dir 'saved_models'
            state = {"model":net.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,
                     "lr":lr.state_dict()}
            torch.save(state, model_dir + model_name)
            print('Finished Training')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)
print(net)
print(next(net.parameters()).device)

batch_size = 50
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
n_epochs = 50
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num,"Trainable":trainable_num}
print(get_parameter_number(net))
#data_dict:{voicename:[{img:_,meant60:_,},{}]}
data_dict = torch.load("same_voice_data_single.pt")
train_net(n_epochs, data_dict,batch_size)
# model_dir = './Checkpoints/relative_loss_addBias/'
# model_name = 'krunal_keypoints_model_1.pt'
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
# torch.save(net.state_dict(), model_dir + model_name)
