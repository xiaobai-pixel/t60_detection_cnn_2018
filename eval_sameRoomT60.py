import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import torch.nn as nn
criterion = nn.SmoothL1Loss()

class ValDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.audio_image_dir = self.root_dir

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):

        val_data = {}
        numpy_image = os.path.join(self.audio_image_dir,
                                   self.key_pts_frame.iloc[idx, 0].split("/")[-2],
                                   self.key_pts_frame.iloc[idx, 0].split("/")[-1])
        image = np.load(numpy_image)
        # #加载图片后，判断该图片是否存在于val_data的关键字中
        # #不存在则要再加一个关键字，存在则添加在原来的关键字那里
        # #file_name_img = 房间名
        #img_name = (numpy_image.split("/")[-1]).split("-")[0] + "-----"   + numpy_image.split("_")[-1].split(".")[0]
        img_name = numpy_image.split("/")[-1]
        img_name_new = img_name.split("_")[1] + "_" + img_name.split("_")[2]
        if img_name_new == "Lecture_Room" or img_name_new =="Meeting_Room" :
            img_name_new = img_name.split("_")[1] + "_" + img_name.split("_")[2] + "_" \
                           + img_name.split("_")[3]
        DDR_each_band = self.key_pts_frame.iloc[idx, 1:31].values
        T60_each_band = self.key_pts_frame.iloc[idx, 61:91].values
        mean_t60 = self.key_pts_frame.iloc[idx, -2:].values
        sample = {'image': image, 'ddr': DDR_each_band, 't60': T60_each_band,'img_name':img_name_new,"meanT60":mean_t60}
        totensor = ToTensor()
        sample = totensor(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, ddr, t60,img_name,mean_t60 = sample['image'], sample['ddr'], sample['t60'],sample["img_name"],sample["meanT60"]
        image = np.expand_dims(image, 0)
        ddr = ddr.astype(float)
        t60 = t60.astype(float)
        mean_t60 = mean_t60.astype(float)
        # image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'ddr': torch.from_numpy(ddr),
                't60': torch.from_numpy(t60),
                "img_name":img_name,
                "meanT60":torch.from_numpy(mean_t60)}


class Val_meanT60():

    def __call__(self,epoch,net,val_loader,lr,device,csv_filename):
        # 因为我要在验证时，对同一段语音的同一通道数下的T60求mean
        # 所以这就需要让你的数据带有四个keys,要包含img_name(音频名字+通道数)
        import numpy as np
        gt_SameRoomT60 = {}  # 同一房间真实的meant60
        pred_SameRoomT60 = {}  # 同一房间预测的meant60
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                # 创建一个字典来存储T60,格式是{str:[Tensor]}

                images = data["image"]
                #t60 = data["t60"]
                # 它是一次加载Batch_size个img
                imgs_name = data["img_name"]
                mean_t60 = data["meanT60"][:,-1]
                mean_t60 = torch.tensor(mean_t60.cpu().detach().numpy(), dtype=torch.float32, device=device)
                images = torch.tensor(images.cpu().detach().numpy(), dtype=torch.float32, device=device)
                output_pts = net(images)[:,-1] #我要取最后一个的T60，因为最后一个才是我想要的
                # 存储真实的meanT60
                gt_mean_t60_idx = 0
                pred_mean_t60_idx = 0
                # room_img_name:[]
                for img_name in imgs_name:
                    if img_name in gt_SameRoomT60.keys():
                        gt_SameRoomT60[img_name].append(mean_t60[gt_mean_t60_idx])
                        gt_mean_t60_idx += 1
                    else:

                        gt_SameRoomT60[img_name] = [mean_t60[gt_mean_t60_idx]]
                        gt_mean_t60_idx += 1

                    # 存储预测的meanT60
                    if img_name in pred_SameRoomT60.keys():
                        pred_SameRoomT60[img_name].append(output_pts[pred_mean_t60_idx])
                        pred_mean_t60_idx += 1
                    else:
                        pred_SameRoomT60[img_name] = [output_pts[pred_mean_t60_idx]]
                        pred_mean_t60_idx += 1
            # 因为数据集是打乱了的，所以你只能先将全部Imgs预测存储，再和真实t60比较
            # 现在开始遍历文件名再开始比较，这样可能会效率比较低，复杂度太高
            #接下来我要对分别对7个房间的预测t60与真实t60做个比较
            #求得每个房间的mse和bias，并用字典存储起来
            bias_all_room = {}
            mse_all_room = {}
            for key in pred_SameRoomT60.keys():
                #分别求针对同一房间的pred_meant60和gt_meant60
                pred_meant60 = pred_SameRoomT60[key]
                gt_meant60 = gt_SameRoomT60[key]
                numOfSamp = len(pred_meant60)
                pred_meant60 = torch.tensor(pred_meant60,dtype=torch.float32,device=device)
                gt_meant60 = torch.tensor(gt_meant60, dtype=torch.float32, device=device)
                bias = torch.sum(pred_meant60 - gt_meant60)/numOfSamp
                square_sum = torch.sum(torch.square(pred_meant60 - gt_meant60))/numOfSamp
                mse  = square_sum
                bias_all_room[key] = bias.item()
                mse_all_room[key] = mse.item()
            eval_file = "./eval_result/relative_loss_addbias/"
            if  not os.path.exists(eval_file) :
                os.makedirs(eval_file)
            txt_name = os.path.join(eval_file,"eval_msebias_byrelativeloss_addbias.txt")
            with open(txt_name,"a") as f:
                f.write(csv_filename)
                f.write("\n")
                f.write("bis is {}".format(bias_all_room))
                f.write("\n")
                f.write("mse is {}".format(mse_all_room))
                f.write("\n")
                f.write("predT60 is {}".format(pred_meant60[:500]))
                f.write("\n")
                f.write("gt_meant60 is {}".format(gt_meant60[:500]))
                f.write("\n")
                f.write("\n")
                f.close()
            return  bias_all_room,mse_all_room