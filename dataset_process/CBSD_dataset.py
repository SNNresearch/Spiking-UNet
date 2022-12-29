import torch
from torch.utils.data import Dataset
import h5py
import os
import torch
from torchvision.transforms import transforms

def load_hdf5(infile):
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]


class test_CBSD_dataset(Dataset):
    def __init__(self, dir_path, file_name, color = True, transform = None):
        super().__init__()
        
        self.data = load_hdf5(os.path.join(dir_path, file_name))
        if color:
            self.label = load_hdf5(os.path.join(dir_path, 'CBSD_original_pictures.hdf5'))
        else:
            self.label = load_hdf5(os.path.join(dir_path, 'BSD_original_pictures.hdf5'))

        self.total_num = self.data.shape[0]
        self.transform = transform

    def __len__(self):
        return self.total_num

    def __getitem__(self, index: int):
        data = self.data[index]
        current_label = self.label[index//4,:,:,:]

        if self.transform is not None:
            data = self.transform(data)
            current_label = self.transform(current_label)
            
        return data, current_label

class train_CBSD_dataset(Dataset):
    def __init__(self, dir_path, color = True, valid = False, transform = None) -> None:
        super().__init__()
        if color:
            self.data = load_hdf5(os.path.join(dir_path, "CBSD_patch_diff_train.hdf5"))
            self.label = load_hdf5(os.path.join(dir_path, "CBSD_patch_diff_label.hdf5"))
        else:
            self.data = load_hdf5(os.path.join(dir_path, "BSD_patch_diff_train.hdf5"))
            self.label = load_hdf5(os.path.join(dir_path, "BSD_patch_diff_label.hdf5"))
        if valid:
            self.data = self.data[10000:-1]
            self.label = self.label[10000:-1]
        else:
            self.data = self.data[:10000]
            self.label = self.label[:10000]
        
        self.total_num = self.data.shape[0]
        self.transform = transform

    def __getitem__(self, index: int):
        data = self.data[index]
        label = self.label[index]
    
        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
        else:
            data = transforms.ToTensor(data)
            label = transforms.ToTensor(label)

        data = data.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return data, label

    def __len__(self) -> int:
        return self.total_num
