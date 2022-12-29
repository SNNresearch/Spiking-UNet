import h5py
import os
import numpy as np
import cv2

def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

def load_hdf5(infile):
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.array(img)
        h, w = img.shape[-2:]
        if h == 481:
            img = img.swapaxes(-2,-1)[...,::-1]
        img = np.expand_dims(img, axis=2)  # HxWx1
        
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        
        img = np.array(img)
        h, w = img.shape[0:-1]
        if h == 481:
            img = img.swapaxes(0,1)
        
    return img



def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def random_crop(img, label, size, num):
    img_patches = []
    label_patches = []
    h_size = size[0]
    w_size = size[1]
    h, w = img.shape[0:-1]
    Ys = np.random.randint(0, h - h_size, num)
    Xs = np.random.randint(0, w - w_size, num)
    for y, x in zip(Ys, Xs):
        img_patch = img[y:(y+h_size), x:(x+w_size), :]
        label_patch = label[y:(y+h_size), x:(x+w_size), :]
        img_patches.append(img_patch)
        label_patches.append(label_patch)
    img_patches = np.array(img_patches)
    label_patches = np.array(label_patches)
    return img_patches, label_patches
