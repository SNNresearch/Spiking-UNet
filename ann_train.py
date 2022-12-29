# type:ignore
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from ann.train import train_and_evaluate
from torch import optim
from model.ann_model.U_Net import *

argparser = argparse.ArgumentParser()
argparser.add_argument('-n',    type=str,   dest='dataset_name',  required=True)
argparser.add_argument('-s1',   type=int,   dest='seed1',         default=0,              required=False)
argparser.add_argument('-s2',   type=int,   dest='seed2',         default=0,              required=False)
argparser.add_argument('-s3',   type=int,   dest='seed3',         default=0,              required=False)
argparser.add_argument('-b',    type=int,   dest='batch_size',    default=8,              required=False)
argparser.add_argument('-e',    type=int,   dest='epochs',        default=100,            required=False)
argparser.add_argument('-op',   type=str,   dest='optim',         default='adam',         required=False)
argparser.add_argument('-lr',   type=float, dest='learning_rate', default=1e-6,           required=False)
argparser.add_argument('-T',    dest='train',  default=False,  action='store_true')


args = argparser.parse_args()

seed1        = args.seed1
seed2        = args.seed2
seed3        = args.seed3

random.seed(seed2)
np.random.seed(seed1)
torch.manual_seed(seed1)
torch.cuda.manual_seed(seed2)
torch.cuda.manual_seed_all(seed3)
cudnn.benchmark = False
cudnn.deterministic = True

epochs         = args.epochs
batch_size     = args.batch_size
dataset_name   = args.dataset_name
learning_rate  = args.learning_rate
train          = args.train
opts           = args.optim

# device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


# 用于测试代码是否跑通: 测试的时候为True
test = False
if dataset_name in ['CBSD', 'BSD', 'Set12']:
    if dataset_name == 'CBSD':
        data_path = './data/benli/CBSD_dataset_new'
        pytorch_model = Denoising_Model(color=True)
    else:
        data_path = './data/benli/BSD_dataset_new'
        pytorch_model = Denoising_Model(color=False)
else:
    if dataset_name == 'CamSeq01':
        data_path = './data/benli/CamSeq01_hdf5'
        pytorch_model = Segmentation_UNet(input_channel=3, class_num=32, fnum=64)
    elif dataset_name == 'ISBI':
        data_path = './data/benli/ISBI_2012'
        pytorch_model = Segmentation_UNet(input_channel=1, class_num=2, fnum=64)
    else:
        data_path = './data/benli/DRIVE'
        pytorch_model = Segmentation_UNet(input_channel=1, class_num=2, fnum=64)

pytorch_model.to(device)

if opts == 'adam':
    optimizer = optim.Adam(pytorch_model.parameters(), lr=learning_rate)
elif opts == 'sgd':
    optimizer = optim.SGD(pytorch_model.parameters(), lr=learning_rate)
elif opts == 'rms':
    optimizer = optim.RMSprop(pytorch_model.parameters(), lr=learning_rate)
elif opts == 'adadelta':
    optimizer = optim.Adaelta(pytorch_model.parameters(), lr=learning_rate)

output_path = './new_results/ann'
train_and_evaluate(pytorch_model, device, data_path, output_path, train, dataset_name, 64, optimizer, epochs=epochs, batch_size=batch_size, test=test)