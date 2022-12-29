from torch.utils.data import Dataset
import os
import torch
import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
import random
import cv2

def load_hdf5(infile):
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]

def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def read_directory(directory_name, label = False):
    img_list = []
    for filename in os.listdir(directory_name):
        img = Image.open(directory_name+'/'+filename).convert('L')
        if label:
            if np.max(img) == 1.0:
                img_np = np.array(img)
            else:
                img_np = np.array(img) / 255.0
            img_np = np.expand_dims(img, axis = 2)
        else:
            img_np = np.array(img) / 255.0
            img_np = np.expand_dims(img, axis = 2)
        img_list.append(img_np)
    image = np.stack(img_list,axis=0)
    return image

def img2patches(imgs, patch_size):
    size = imgs.shape[0]
    H, W = imgs.shape[1:3]
    H_patch_num = H // patch_size
    W_patch_num = W // patch_size
    img_patches_list = []
    for index in range(size):
        for H_patch in range(H_patch_num):
            H_start = H_patch * patch_size
            H_end = (H_patch + 1) * patch_size
            for W_patch in range (W_patch_num):
                W_start = W_patch * patch_size
                W_end = (W_patch + 1) * patch_size
                img_patches_list.append(imgs[index, H_start:H_end, W_start:W_end,:])
    img_patches = np.stack(img_patches_list, axis = 0)
    return img_patches

def patch2img(img_patch, label):
    h, w = label.shape
    img = np.zeros_like(label)
    img[:h//2,:w//2] = img_patch[0]
    img[:h//2,w//2:w] = img_patch[1]
    img[h//2:h,:w//2] = img_patch[2]
    img[h//2:h,w//2:w] = img_patch[3]

    return img





class train_ISBI_dataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'data/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        # label_path = image_path.replace('data', 'label')
        index = image_path.rfind('data')
        label_path = image_path[:index] + image_path[index:].replace('data', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

class test_ISBI_dataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'data/*.png'))
    
    def __getitem__(self, index: int):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        # label_path = image_path.replace('data', 'label')
        index = image_path.rfind('data')
        label_path = image_path[:index] + image_path[index:].replace('data', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        return image, label


    def __len__(self) -> int:
        return len(self.imgs_path)


