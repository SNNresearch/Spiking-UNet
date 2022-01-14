import torch
import torch.nn as nn
import argparse
from snn.spiking_neuron import IF_Neuron_layer_vth, IF_Neuron_asf_bp
import os
import numpy as np
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from ann_snn_segmentation import params_dict
from snn.conversion.utils import Parser
from dataset_process.CamSeq_dataset import CamSeq_dataset, load_hdf5, onehot_to_rgb

import numpy as np
from dataset_process.ISBI_dataset import train_ISBI_dataset, test_ISBI_dataset
from dataset_process.DRIVE_dataset import DRIVE_dataset
import configparser   
from tools.file_tools import check_exist_makedirs
from model.ann_model.U_Net import parallel_v2
import random

os.environ['PYTHONWARNINGS'] = 'ignore'

class Segmentation_UNet_shallow_v2(nn.Module):
    def __init__(self, input_channel = 1, class_num = 2, fnum = 64, reset_method='reset_by_subtraction',device='cpu', timesteps=100):
        super(Segmentation_UNet_shallow_v2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = input_channel, out_channels = fnum, kernel_size = 3, padding = 1, bias = True)
        # self.relu1 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps)
        self.relu1 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.conv2 = nn.Conv2d(in_channels = fnum, out_channels = fnum, kernel_size = 3, padding = 1, bias = True)
        # self.relu2 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps)
        self.relu2 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.pooling1 = nn.AvgPool2d(kernel_size = 2, padding = 0)
        # self.pooling1 = nn.MaxPool2d(kernel_size = 2, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = fnum, out_channels = fnum * 2, kernel_size = 3, padding = 1, bias = True)
        # self.relu3 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps)
        self.relu3 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.conv4 = nn.Conv2d(in_channels = fnum * 2, out_channels = fnum * 2, kernel_size = 3, padding = 1, bias = True)
        # self.relu4 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps)
        self.relu4 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.pooling2 = nn.AvgPool2d(kernel_size = 2, padding = 0)
        # self.pooling2 = nn.MaxPool2d(kernel_size = 2, padding = 0)
        self.conv5 = nn.Conv2d(in_channels = fnum * 2, out_channels = fnum * 4, kernel_size = 3, padding = 1, bias = True)
        # self.relu5 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps)
        self.relu5 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.conv6 = nn.Conv2d(in_channels = fnum * 4, out_channels = fnum * 4, kernel_size = 3, padding = 1, bias = True)
        # self.relu6 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps)
        self.relu6 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.convtranspose1 = nn.ConvTranspose2d(in_channels = fnum * 4, out_channels = fnum * 2, stride = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        # self.relu_convtranpose1 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps)
        self.relu_convtranpose1 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.conv7 = nn.Conv2d(in_channels = fnum * 4, out_channels = fnum * 2, kernel_size = 3, padding = 1, bias = True)
        # self.relu7 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps) 
        self.relu7 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.conv8 = nn.Conv2d(in_channels = fnum * 2, out_channels = fnum * 2, kernel_size = 3, padding = 1, bias = True)
        # self.relu8 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps)
        self.relu8 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.convtranspose2 = nn.ConvTranspose2d(in_channels = fnum * 2, out_channels = fnum, stride = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        # self.relu_convtranpose2 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps) 
        self.relu_convtranpose2 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.conv9 = nn.Conv2d(in_channels = fnum * 2, out_channels = fnum, kernel_size = 3, padding = 1, bias = True)
        # self.relu9 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps)
        self.relu9 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.conv10 = nn.Conv2d(in_channels = fnum, out_channels = fnum, kernel_size = 3, padding = 1, bias = True)
        # self.relu10 = IF_Neuron_layer_vth(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', reduce_error=False, optimizer_vth=False, device=device, avg=False, timesteps=timesteps)
        self.relu10 = IF_Neuron_asf_bp(v_threshold=1.0, v_reset=0.0, reset_method=reset_method, state='eval', device=device)
        self.conv11 = nn.Conv2d(in_channels = fnum, out_channels = class_num, kernel_size = 1, bias = True)
    
    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.relu1(x_1)
        x_1 = self.conv2(x_1)
        x_1 = self.relu2(x_1)
        down_x1 = self.pooling1(x_1)
        x_2 = self.conv3(down_x1)
        x_2 = self.relu3(x_2)
        x_2 = self.conv4(x_2)
        x_2 = self.relu4(x_2)
        down_x2 = self.pooling2(x_2)
        x_3 = self.conv5(down_x2)
        x_3 = self.relu5(x_3)
        x_3 = self.conv6(x_3)
        x_3 = self.relu6(x_3)
        up1 = self.convtranspose1(x_3)
        up1 = self.relu_convtranpose1(up1)
        concat1 = torch.cat((x_2, up1), dim = 1)
        x_4 = self.conv7(concat1)
        x_4 = self.relu7(x_4)
        x_4 = self.conv8(x_4)
        x_4 = self.relu8(x_4)
        up2 = self.convtranspose2(x_4)
        up2 = self.relu_convtranpose2(up2)
        concat2 = torch.cat((x_1, up2), dim = 1)
        x_5 = self.conv9(concat2)
        x_5 = self.relu9(x_5)
        x_5 = self.conv10(x_5)
        x_5 = self.relu10(x_5)
        output = self.conv11(x_5)
        return output

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s1', type=int, dest='seed1', required=True)
    argparser.add_argument('-s2', type=int, dest='seed2', required=True)
    argparser.add_argument('-name', type=str, dest='dataset_name', required=True)
    # argparser.add_argument('-s3', type=int, dest='seed3', required=True)
    argparser.add_argument('-b', type=int, dest='batch_size', required=True)
    argparser.add_argument('-t', type=int, dest='timesteps', required=True)
    argparser.add_argument('-v', type=str, choices=['original', 'v1', 'v2', 'v3'], dest='version', required=True)
    argparser.add_argument('--cfg', type=str, default=None, dest='config_path', required=False)
    argparser.add_argument('-th', type=float, default=1.0, dest='threshold', required=False)
    argparser.add_argument('-r', type=str, choices=['zero','sub'], dest='reset_method',required=True)
    argparser.add_argument('-nc', type=int, choices=[32, 64], dest='network_input_channels', required=True)
    argparser.add_argument('-o', type=str, choices=['mem', 'spike'], dest='output_format', required=True)
    argparser.add_argument('-lr', type=float, dest='learning_rate', required=True)
    argparser.add_argument('-op', type=str, choices=['adam', 'RMS'], dest='optimizer_class', required=True)
    argparser.add_argument('-e', type=int, dest='epochs', required=True)
    argparser.add_argument('-tm', type=str, dest='train_method', required=True)

    args = argparser.parse_args()

    if args.config_path is not None:
        config_path = args.config_path

    # gpu = args.gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu 

    seed1 = args.seed1
    seed2 = args.seed2
    # seed3 = args.seed3

    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed2)
    # torch.cuda.manual_seed_all(seed3)
    np.random.seed(seed1)
    random.seed(seed2)
    cudnn.benchmark = False
    cudnn.deterministic = True

    dataset_name = args.dataset_name
    batch_size = args.batch_size
    train_method = args.train_method
    version = args.version 
    timesteps = args.timesteps
    threshold = args.threshold
    reset = args.reset_method
    output_format = args.output_format
    learning_rate = args.learning_rate
    optimizer_class = args.optimizer_class
    epochs = args.epochs

    if reset == 'zero':
        reset_method = 'reset_by_zero'
    elif reset == 'sub':
        reset_method = 'reset_by_subtraction'

    network_input_channels = args.network_input_channels

    # color = 'RGB'
    if dataset_name == 'CamSeq01':
        color = 'RGB'
        data_path = '/data/benli/CamSeq01_hdf5'
    else:
        color = 'GRAY'
        if dataset_name == 'ISBI':
            data_path = '/data/benli/ISBI_2012'
        else:
            data_path = '/data/benli/DRIVE_dataset'
    task = 'segmentation'
    base_path = os.path.join('/output', 'snn', dataset_name)
    save_path = os.path.join(base_path, reset_method, 
        version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_steps' + str(timesteps) + '_train',
        's1_'+ str(seed1) + '_s2_' + str(seed2) + '_batch_size' + str(batch_size) + '_lr' + str(learning_rate) + '_ep' + str(epochs) + '_' + optimizer_class + '_' + train_method
    )
    config_path = os.path.join(save_path, 'config.txt')
    log_dir = os.path.join('/output', 'logs', 'snn', 
        dataset_name, 
        reset_method,
        version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_train',
        's1_'+ str(seed1) + '_s2_' + str(seed2) + '_batch_size' + str(batch_size) + '_steps' + str(timesteps) + '_lr' + str(learning_rate) + '_ep' + str(epochs) + '_' + optimizer_class + '_' + train_method
    )
    model_save = os.path.join('/output/snn_model', dataset_name, str(network_input_channels))
    model_save_path = os.path.join(model_save, 's1_'+ str(seed1) + '_s2_' + str(seed2) + '_batch_size' + str(batch_size) + '_steps' + str(timesteps) + '_' + train_method + '_' + version + '.pth')

    check_exist_makedirs(model_save)
    check_exist_makedirs(save_path)
    check_exist_makedirs(log_dir)

    if color == 'RGB':
        input_channel = 3
    else:
        input_channel = 1
    if dataset_name == 'CamSeq01':
        output_channel = 32
    else:
        output_channel = 2
    
    # if optimizer_method == 'stbp':
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # else:
    # device = torch.device(('cuda:' + gpu) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # model
    snn_model = Segmentation_UNet_shallow_v2(input_channel=input_channel, class_num=output_channel, fnum=network_input_channels, reset_method=reset_method, device=device, timesteps=timesteps) 
    snn_model = snn_model.to(device)

    if optimizer_class == 'adam':
        optimizer = optim.Adam(snn_model.parameters(), lr=learning_rate)
    elif optimizer_class == 'RMS':
        optimizer = optim.RMSprop(snn_model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)

    train = True
    from snn.asf_bp_train import train_and_evaluate
    train_and_evaluate(
        snn_model, device, "no_op_vth",
        train, data_path, log_dir, 
        model_save_path, save_path, 
        dataset_name, timesteps, output_format, 
        optimizer, epochs=epochs, batch_size=batch_size)

    # config write
    config = configparser.ConfigParser()
    config['path'] = {
        'data_path': data_path,
        'snn_model_path': model_save_path
    }
    config['DEFAULT'] = {
        'dataset': dataset_name,
        'task': task,
        'color': str(color),
        'batch_size': str(batch_size),
        'network_size': version,
        'network_input_channels': str(network_input_channels)
    }

    config['SNN setting'] = {
        'train': train,
        'epoch': epochs,
        'reset_method': reset_method,
        'timesteps': timesteps,
        'optimizer': optimizer_class,
        'learning_rate': learning_rate
    }

    with open(config_path, 'w') as configfile:
        config.write(configfile)