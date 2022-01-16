import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import torchvision.transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from torchvision import transforms, utils


class saveSameVoice(Dataset):
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

    #暴力扫描数据，并进行存储
    def ergodic_data(self):
        same_voice_data = {}
        print(self.key_pts_frame.shape[0])
        print("start saving data")
        for idx in range(self.key_pts_frame.shape[0]):

            #先转换数据格式
            numpy_image = os.path.join(self.audio_image_dir,
                                       self.key_pts_frame.iloc[idx, 0].split("/")[-2],
                                       self.key_pts_frame.iloc[idx, 0].split("/")[-1])
            image = np.load(numpy_image)
            # key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
            DDR_each_band = self.key_pts_frame.iloc[idx, 1:31].values
            T60_each_band = self.key_pts_frame.iloc[idx, 61:91].values
            MeanT60_each_band = self.key_pts_frame.iloc[idx, -2:].values
            sample = {'image': image, 'ddr': DDR_each_band, 't60': T60_each_band, "MeanT60": MeanT60_each_band}
            if self.transform:
                sample = self.transform(sample)
            #之后用字典的形式存储同一语音下的数据
            name_img = self.key_pts_frame.iloc[idx, 0]
            same_voice_name = (name_img.split("/")[-1]).split("-")[0]
            if same_voice_name in same_voice_data.keys():
                same_voice_data[same_voice_name].append(sample)
            else:
                same_voice_data[same_voice_name] = [sample]
        torch.save(same_voice_data,"same_voice_data_single.pt")
        print("finished save")
        return same_voice_data


# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # scale color range from [0, 255] to [0, 1]
        image_copy = image_copy / 255.0

        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100) / 50.0

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, key_pts = sample['image'], sample['keypoints']
        #
        # # if image has no grayscale color channel, add one
        # if(len(image.shape) == 2):
        #     # add that third color dim
        #     image = image.reshape(image.shape[0], image.shape[1], 1)
        #
        # # swap color axis because
        # # numpy image: H x W x C
        # # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        #
        # return {'image': torch.from_numpy(image),
        #         'keypoints': torch.from_numpy(key_pts)}

        image, ddr, t60, meanT60 = sample['image'], sample['ddr'], sample['t60'], sample["MeanT60"]
        image = np.expand_dims(image, 0)
        ddr = ddr.astype(float)
        t60 = t60.astype(float)
        meanT60 = meanT60.astype(float)
        # image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'ddr': torch.from_numpy(ddr),
                't60': torch.from_numpy(t60),
                "MeanT60": torch.from_numpy(meanT60)}







data_transform = transforms.Compose([ToTensor()])

transformed_dataset = saveSameVoice(csv_file='/data2/queenie/IEEE2015Ace/solution/DatasetProcessing/Single.csv',
                                             root_dir='/data2/queenie/IEEE2015Ace/solution/DatasetProcessing/',
                                             transform=data_transform)
data = transformed_dataset.ergodic_data()

