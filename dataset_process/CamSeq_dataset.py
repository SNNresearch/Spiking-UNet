from torch.utils.data import Dataset
import os
import torch
import h5py
import numpy as np
from PIL import Image
import glob
import random
import cv2

def load_hdf5(infile):
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]

def parse_code(l):
    '''Function to parse lines in a text file, returns separated elements (label codes and names in this case)
    '''
    if len(l.strip().split("\t")) == 2:
        a, b = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), b
    else:
        a, b, c = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), c

def rgb_to_onehot(rgb_image, colormap):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    output = np.zeros(onehot.shape + (3,))

    for k in colormap.keys():
        output[onehot==k] = colormap[k]
        
    return np.uint8(output)

class CamSeq_dataset(Dataset):
    def __init__(self, path, train = True) -> None:
        super().__init__()

        if train:
            self.data = load_hdf5(os.path.join(path, 'CamSeq01_dataset_imgs_train.hdf5'))
            self.label = load_hdf5(os.path.join(path, 'CamSeq01_dataset_groundTruth_train.hdf5'))
        else:
            self.data = load_hdf5(os.path.join(path, 'CamSeq01_dataset_imgs_test.hdf5'))
            self.label = load_hdf5(os.path.join(path, 'CamSeq01_dataset_groundTruth_test.hdf5'))
        
        label_codes, label_names = zip(*[parse_code(l) for l in open(os.path.join(path, "label_colors.txt"))])
        label_codes, label_names = list(label_codes), list(label_names)
        self.code2id = {v:k for k,v in enumerate(label_codes)}
        self.id2code = {k:v for k,v in enumerate(label_codes)}
        self.name2id = {v:k for k,v in enumerate(label_names)}
        self.id2name = {k:v for k,v in enumerate(label_names)}

        assert(self.data.shape == self.label.shape)
        self.total_num = self.data.shape[0]
    
    def __getitem__(self, index: int):
        data = self.data[index] / 255.0
        rgb_label = self.label[index]
        one_hot_label = rgb_to_onehot(rgb_label, self.id2code)
        data = np.transpose(data,(2,0,1))
        one_hot_label = np.argmax(one_hot_label, axis=2)

        return data, one_hot_label

    def __len__(self) -> int:
        return self.total_num

        
