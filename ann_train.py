import os
import torch
import shutil
import argparse
import configparser
from tools.file_tools import check_exist_makedirs
import numpy as np
from torch import optim
from ann.train import train_and_evaluate
import random
import torch.backends.cudnn as cudnn

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, dest="config_path", required=True)
    args = argparser.parse_args()
    config_path = args.config_path
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # torch.cuda.manual_seed_all(seed3)
    np.random.seed(0)
    random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    dataset_name = config.get("path", "dataset")
    if dataset_name == 'CamSeq01':
        class_num = 32
    else:
        class_num = 2

    color = config.get("DEFAULT", "color")
    if color == 'RGB':
        input_channel = 3
    else:
        input_channel = 1

    network_input_channels = config.getint("DEFAULT", "network_input_channels")
    task = config.get("DEFAULT", "task")
    network_size = config.get("DEFAULT", "network_size")
    if task == 'segmentation':
        if network_size == 'original':
            from model.ann_model.U_Net import Segmentation_UNet
            net = Segmentation_UNet(input_channel=input_channel, class_num=class_num, fnum=network_input_channels)
        elif network_size == 'v1':
            from model.ann_model.U_Net import Segmentation_UNet_shallow_v1
            net = Segmentation_UNet_shallow_v1(input_channel=input_channel, class_num=class_num, fnum=network_input_channels)
        elif network_size == 'v2':
            from model.ann_model.U_Net import Segmentation_UNet_shallow_v2
            net = Segmentation_UNet_shallow_v2(input_channel=input_channel, class_num=class_num, fnum=network_input_channels)
        elif network_size == 'v3':
            from model.ann_model.U_Net import Segmentation_UNet_shallow_v3
            net = Segmentation_UNet_shallow_v3(input_channel=input_channel, class_num=class_num, fnum=network_input_channels)
    

    # 加载训练参数和优化器
    epochs = config.getint("ANN setting", "epoch")
    learning_rate = config.getfloat("ANN setting", "learning_rate")
    batch_size = config.getint("DEFAULT", "batch_size")
    optimizer_class = config.get("ANN setting", "optimizer")
    if optimizer_class == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer_class == 'RMS':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)

    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = config.get('path', 'data_path')
    output_path = './output'
    config_copy_path = os.path.join(output_path,'ann', dataset_name, network_size, str(network_input_channels))
    check_exist_makedirs(config_copy_path)
    shutil.copy(config_path, config_copy_path) 

    method = config.get('ANN setting', 'method')
    if method == 'train':
        train = True
    else:
        train = False

    train_and_evaluate(net, device, data_path, output_path, train = train, dataset_name=dataset_name, version=network_size, input_channels=network_input_channels, optimizer=optimizer, epochs=epochs, batch_size=batch_size)
