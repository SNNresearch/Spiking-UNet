from numpy.lib.type_check import imag
from torch.utils import data
from torch.utils.data import Dataset
from dataset_process.ISBI_dataset import load_hdf5
import glob
import os 
import numpy as np
import cv2
import copy

class Set12_Dataset(Dataset):
    def __init__(self, data_path, noise_level):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.png'))
        self.noise_level = noise_level

    def __getitem__(self, index: int):
        np.random.seed(seed=0)
        image_path = self.imgs_path[index] 
        label = cv2.imread(image_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = np.asarray(label) / 255.
        label = np.expand_dims(label, axis = 0)
        image = copy.deepcopy(label)

        add_noise = np.random.normal(0, self.noise_level / 255., image.shape)
        image = image + add_noise   
        

        return image, label
    
    def __len__(self) -> int:
        return len(self.imgs_path)

class Set12_Dataset_hdf5(Dataset):
    def __init__(self, data_path, noise_level):
        if noise_level == 15:
            self.images_256 = load_hdf5(os.path.join(data_path, 'Set_test_256_15.hdf5'))
            self.images_512 = load_hdf5(os.path.join(data_path, 'Set_test_512_15.hdf5'))
        elif noise_level == 25:
            self.images_256 = load_hdf5(os.path.join(data_path, 'Set_test_256_25.hdf5'))
            self.images_512 = load_hdf5(os.path.join(data_path, 'Set_test_512_25.hdf5'))
        elif noise_level == 35:
            self.images_256 = load_hdf5(os.path.join(data_path, 'Set_test_256_35.hdf5'))
            self.images_512 = load_hdf5(os.path.join(data_path, 'Set_test_512_35.hdf5'))
        elif noise_level == 45:
            self.images_256 = load_hdf5(os.path.join(data_path, 'Set_test_256_45.hdf5'))
            self.images_512 = load_hdf5(os.path.join(data_path, 'Set_test_512_45.hdf5'))
        elif noise_level == 50:
            self.images_256 = load_hdf5(os.path.join(data_path, 'Set_test_256_50.hdf5'))
            self.images_512 = load_hdf5(os.path.join(data_path, 'Set_test_512_50.hdf5'))

        self.label_256 = load_hdf5(os.path.join(data_path, 'Set_test_256_orignal.hdf5'))
        self.label_512 = load_hdf5(os.path.join(data_path, 'Set_test_512_orignal.hdf5'))
        self.len_256 = self.images_256.shape[0]
        self.total_len = self.images_256.shape[0] + self.images_512.shape[0]

    def __getitem__(self, index: int):
        if index < self.len_256:
            image = self.images_256[index]
            label = self.label_256[index]
        else:
            image = self.images_512[index - self.len_256]
            label = self.label_512[index - self.len_256]

        return image,label
    
    def __len__(self) -> int:
        return self.total_len
