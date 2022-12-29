from torch.utils.data import Dataset
import os
import sys
sys.path.append('../')
import numpy as np
# import random
import h5py
import snn.tools.segmentation.extract_patches as ep

get_data_training = ep.get_data_training
get_data_testing = ep.get_data_testing

def load_hdf5(infile):
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]

class DRIVE_dataset(Dataset):
    def __init__(self, data_path, train):
        super().__init__()
        self.train = train
        if self.train:
            self.patches_imgs_train, self.patches_masks_train = get_data_training(  DRIVE_train_imgs_original=os.path.join(data_path, 'DRIVE_dataset_imgs_train.hdf5'),
                                                                                    DRIVE_train_groudTruth=os.path.join(data_path, 'DRIVE_dataset_groundTruth_train.hdf5'),  #masks
                                                                                    patch_height=48,
                                                                                    patch_width=48,
                                                                                    N_subimgs=12000,
                                                                                    inside_FOV=False)
            self.total_num = self.patches_imgs_train.shape[0]
        else:
            self.patches_imgs_test, self.patches_masks_test = get_data_testing(
                                                        DRIVE_test_imgs_original=os.path.join(data_path, 'DRIVE_dataset_imgs_test.hdf5'),  #original
                                                        DRIVE_test_groudTruth=os.path.join(data_path, 'DRIVE_dataset_groundTruth_test.hdf5'),  #masks
                                                        Imgs_to_test=20,
                                                        patch_height=48,
                                                        patch_width=48,
                                                    )
            self.total_num = self.patches_imgs_test.shape[0]
        
    def __getitem__(self, index: int):
        if self.train:
            image = self.patches_imgs_train[index]
            label = self.patches_masks_train[index]
        else:
            image = self.patches_imgs_test[index]
            label = self.patches_masks_test[index]
        image = np.transpose(image, (2,0,1))
        label = np.squeeze(label, axis = 2)
        

        return image, label

    def __len__(self) -> int:
        return self.total_num
