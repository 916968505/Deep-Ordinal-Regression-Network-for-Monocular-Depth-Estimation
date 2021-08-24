import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import h5py
from PIL import Image
import numpy as np
from utils import load_split

data_path = './data/nyu_depth_v2_labeled.mat'
#data_path = '/home/caojinghao/dorn_depth_estimation/DataSet//home/caojinghao/dorn_depth_estimation/DataSet/nyu_depth_v2_labeled.mat'
batch_size = 2
iheight, iwidth = 480, 640 # raw image size
alpha, beta = 0.02, 10.02
K = 68
output_size = (257, 353)

# 加载NYU_mat类型数据集
class NYU_Dataset(data.Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists
        self.nyu = h5py.File(self.data_path)
        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']
        self.output_size = (257, 353)

    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(2, 1, 0) #HWC
        dpt = self.dpts[img_idx].transpose(1, 0)
        img = Image.fromarray(img)
        dpt = Image.fromarray(dpt)
        img_transform = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor()
        ])
        dpt_transform = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor()
        ])
        img = img_transform(img)
        dpt = dpt_transform(dpt)
        # 将深度图变化到对数空间
        dpt = get_depth_log(dpt)
        return img, dpt

    def __len__(self):
        return len(self.lists)


#从(0,K)->(alpha, beta)
def get_depth_log(depth):
    alpha_ = torch.FloatTensor([alpha])
    beta_ = torch.FloatTensor([beta])
    K_ = torch.FloatTensor([K])
    t = K_ * torch.log(depth / alpha_) / torch.log(beta_ / alpha_)
    # t = t.int()
    return t

# 从(alpha,beta)->(0,K)
def get_depth_sid(depth_labels):
    depth_labels = depth_labels.data.cpu()
    alpha_ = torch.FloatTensor([alpha])
    beta_ = torch.FloatTensor([beta])
    K_ = torch.FloatTensor([K])
    t = torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * depth_labels / K_)
    return t

def getNYUDataset():
    train_lists, val_lists, test_lists = load_split()

    train_set = NYU_Dataset(data_path=data_path, lists=train_lists)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    val_set = NYU_Dataset(data_path=data_path, lists=val_lists)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)

    test_set = NYU_Dataset(data_path=data_path, lists=test_lists)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader

