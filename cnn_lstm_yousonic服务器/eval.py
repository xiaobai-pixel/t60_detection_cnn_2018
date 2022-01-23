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
from models import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
from eval_sameRoomT60 import ValDataset,Val_meanT60
import torch.optim as optim
import datetime

def eval_net(n_epochs, val_loader,csv_filename):
    # prepare the net for training
    net.train()
    # 加载预训练权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("./Checkpoints/relative_loss_addBias/t60_predict_model_93_meanT60_continue11.pt", map_location=device)
    #
    print(net.load_state_dict(checkpoint["model"], strict=False))
    print("begin eval")
    val = Val_meanT60()
    bias,mse = val(1, net, val_loader,0.001,device,csv_filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)
print(net)
print(next(net.parameters()).device)
data_transform = transforms.Compose([ToTensor()])
batch_size = 1024*8
csv_filenames = ["Chromebook.csv","Crucif.csv","Lin8Ch.csv","Mobile.csv","Single.csv","EM32.csv"]
# csv_filenames = ["Mobile.csv","Single.csv","Crucif.csv","Chromebook.csv"]
for csv_filename in csv_filenames:
    csv_file = os.path.join('/data2/queenie/IEEE2015Ace/solution/DatasetProcessing_eval',csv_filename)
    val_dataset = ValDataset(csv_file= csv_file,
                             root_dir='/data2/queenie/IEEE2015Ace/solution/DatasetProcessing_eval/',
                             transform=None)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0, drop_last=True)
    n_epochs = 1  # start small, and increase when you've decided on your model structure and hyperparams
    eval_net(n_epochs, val_loader,csv_filename)
    print("validation finished")




# val_dataset = ValDataset(csv_file='/data2/queenie/IEEE2015Ace/solution/DatasetProcessing_eval/Single.csv',
#                          root_dir='/data2/queenie/IEEE2015Ace/solution/DatasetProcessing_eval/',
#                          transform=None)
# val_loader = DataLoader(val_dataset,
#                         batch_size=batch_size,
#                         shuffle=True,
#                         num_workers=0, drop_last=True)
# n_epochs = 1  # start small, and increase when you've decided on your model structure and hyperparams
# train_net(n_epochs, val_loader)
# print("validation finished")


