# type:ignore
import os
import torch
import random
import argparse
import numpy as np
from snn import simulation
from snn.conversion import Parser
from model.ann_model.U_Net import Denoising_Model, Segmentation_UNet

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_process.CBSD_dataset import load_hdf5, test_CBSD_dataset
from dataset_process.Set_dataset import Set12_Dataset_hdf5
from dataset_process.DRIVE_dataset import DRIVE_dataset
from dataset_process.ISBI_dataset import train_ISBI_dataset, test_ISBI_dataset
from dataset_process.CamSeq_dataset import CamSeq_dataset


def evaluate(snn_model, base_path, dataset_name, timesteps, neuron_class, vth, record_timestep, reset_method, scale_method, method, device, test, **kwargs):

    if dataset_name in ['CBSD', 'BSD', 'Set12']:
        noise_level = kwargs['noise_level']
        kwargs = {}
        if neuron_class == 'one': 
            path = os.path.join(base_path, scale_method, reset_method, dataset_name, method, 'sigma' + str(noise_level) + '_t' + str(timesteps) + '_neuron_' + neuron_class + '_vth' + str(vth))
            logs_path = os.path.join(base_path , 'logs', scale_method, reset_method, dataset_name, method, 'sigma' + str(noise_level) + '_timesteps' + str(timesteps) + '_neuron_' + neuron_class + '_vth' + str(vth))
        else:
            path = os.path.join(base_path, scale_method, reset_method, dataset_name, method, 'sigma' + str(noise_level) + '_t' + str(timesteps) + '_neuron_' + neuron_class)
            logs_path = os.path.join(base_path , 'logs', scale_method, reset_method, dataset_name, method, 'sigma' + str(noise_level) + '_timesteps' + str(timesteps) + '_neuron_' + neuron_class)
        if dataset_name == "CBSD":
            test_dataset = test_CBSD_dataset(dir_path='./data/benli/CBSD_dataset_new', file_name='CBSD_patch_test_img_sigma_'+ str(noise_level) + '.hdf5', color=True, transform=transforms.ToTensor())
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        elif dataset_name == "BSD":
            test_dataset = test_CBSD_dataset(dir_path='./data/benli/BSD_dataset_new', file_name='BSD_patch_test_img_sigma_'+ str(noise_level) + '.hdf5', color=False, transform=transforms.ToTensor())
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        else:
            test_dataset = Set12_Dataset_hdf5(data_path='./data/benli/Set12_new/test', noise_level=noise_level)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        if neuron_class == 'one':
            path = os.path.join(base_path, scale_method, reset_method, dataset_name, method, 't' + str(timesteps) + '_neuron_' + neuron_class + '_vth' + str(vth))
            logs_path = os.path.join(base_path , 'logs', scale_method, reset_method,  dataset_name, method, 'timesteps' + str(timesteps) + '_neuron_' + neuron_class + '_vth' + str(vth))
        else:
            path = os.path.join(base_path, scale_method, reset_method, dataset_name, method, 't' + str(timesteps) + '_neuron_' + neuron_class)
            logs_path = os.path.join(base_path , 'logs', scale_method, reset_method,  dataset_name, method, 'timesteps' + str(timesteps) + '_neuron_' + neuron_class)
        if dataset_name == 'DRIVE':
            border_masks = load_hdf5('./data/benli/DRIVE/DRIVE_dataset_borderMasks_test.hdf5')
            test_dataset = DRIVE_dataset(data_path='./data/benli/DRIVE', train=False)
            test_dataloader = DataLoader(test_dataset, batch_size=156)
            kwargs = {'border_masks' : border_masks}
        elif dataset_name == 'ISBI':
            test_dataset = test_ISBI_dataset('./data/benli/ISBI_2012/test')
            test_dataloader = DataLoader(test_dataset, batch_size=1)
            kwargs = {}
        else:
            test_dataset = CamSeq_dataset('./data/benli/CamSeq01_hdf5', train=False)
            test_dataloader = DataLoader(test_dataset, batch_size=1)
            kwargs = {'id2code': test_dataset.id2code}

    snn_simulator = simulation.simulator(timesteps=timesteps, path=path, logs_path=logs_path, device=device, dataset_name=dataset_name, **kwargs)
    snn_simulator.simulate_mem(snn_model, test_dataloader, eval_train=False, record_timestep=record_timestep, test=test)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n',      type = str,        dest    = 'dataset_name', required = True)
    argparser.add_argument('-t',      type = int,        dest    = 'timesteps',    required = True)
    argparser.add_argument('-m',      type = str,        dest    = 'method',       default  = 'connection_wise', required  = False)
    argparser.add_argument('-s',      type = str,        dest    = 'scale_method', default  = 'robust',       required  = False)
    argparser.add_argument('-neuron', type = str,        dest    = 'neuron_class', default  = 'multi',        required  = False)
    argparser.add_argument('-vth',    type = float,      dest    = 'vthreshold',   default  = 1.0,            required  = False)
    argparser.add_argument('-path',   type = str,        dest    = 'base_path',    default  = './new_results/seg_conversion', required=False)
    argparser.add_argument('-d',      dest = 'sub',      default = False,          action   = 'store_true')
    argparser.add_argument('-record', dest = 'record',   default = False,          action   = 'store_true')

    args = argparser.parse_args()

    # noise_level = args.noise_level
    # gpu = args.gpu
    neuron_class = args.neuron_class
    dataset_name = args.dataset_name
    method       = args.method
    timesteps    = args.timesteps
    scale_method = args.scale_method
    sub          = args.sub
    record       = args.record
    vth          = args.vthreshold
    base_path    = args.base_path


    if sub:
        reset_method = 'reset_by_subtraction'
    else:
        reset_method = 'reset_by_zero'


    random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test=True
    if dataset_name in ['CBSD', 'BSD', 'Set12']:
        if dataset_name == 'CBSD':
            train_data = load_hdf5(os.path.join('./data/benli/CBSD_dataset/CBSD_patch_diff_train.hdf5'))
            pytorch_model = Denoising_Model(color=True)
            pytorch_model.load_state_dict(torch.load('./model/benli/CBSD/CBSD.pth', map_location='cpu'))
            parser = Parser(path = './lambda_factor/CBSD')
        else:
            train_data = load_hdf5(os.path.join('./data/benli/BSD_dataset/BSD_patch_diff_train.hdf5'))
            pytorch_model = Denoising_Model(color=False)
            pytorch_model.load_state_dict(torch.load('./model/benli/BSD/BSD.pth', map_location='cpu'))
            parser = Parser(path = './lambda_factor/BSD')

        L=[random.randint(0,train_data.shape[0] - 1) for _ in range(32)]
        norm_data = train_data[L,:,:,:] 
        norm_data = np.transpose(norm_data, (0,3,1,2))
        norm_data_tensor = torch.from_numpy(norm_data)
        norm_data_tensor = norm_data_tensor.type(torch.FloatTensor)
        parser_model = parser.parse(pytorch_model,norm_data_tensor, method=method, scale_method=scale_method)

        snn_model = parser.convert_to_snn(parser_model, neuron_class=neuron_class, timesteps=timesteps, reset_method=reset_method, v_threshold=vth)
        snn_model.to(device)

        kwargs = {'noise_level': 15}
        evaluate(snn_model, base_path=base_path, dataset_name=dataset_name, timesteps=timesteps, neuron_class=neuron_class, vth=vth, record_timestep=record,
            reset_method=reset_method, scale_method=scale_method, method=method, device=device, test=test, **kwargs)
        kwargs = {'noise_level': 25}
        evaluate(snn_model, base_path=base_path, dataset_name=dataset_name, timesteps=timesteps, neuron_class=neuron_class, vth=vth, record_timestep=record,
            reset_method=reset_method, scale_method=scale_method, method=method, device=device, test=test, **kwargs)
        kwargs = {'noise_level': 35}
        evaluate(snn_model, base_path=base_path, dataset_name=dataset_name, timesteps=timesteps, neuron_class=neuron_class, vth=vth, record_timestep=record,
            reset_method=reset_method, scale_method=scale_method, method=method, device=device, test=test, **kwargs)
    else:
        if dataset_name == 'DRIVE':
            train_dataset = DRIVE_dataset('./data/benli/DRIVE', train=True)
            pytorch_model = Segmentation_UNet(input_channel=1, class_num=2, fnum=64)
            pytorch_model.load_state_dict(torch.load('./model/benli/DRIVE/DRIVE.pth', map_location='cpu'))
            parser = Parser(path = './lambda_factor/DRIVE')
        elif dataset_name == 'ISBI':
            train_dataset = train_ISBI_dataset('./data/benli/ISBI_2012/train')
            pytorch_model = Segmentation_UNet(input_channel=1, class_num=2, fnum=64)
            pytorch_model.load_state_dict(torch.load('./model/benli/ISBI/ISBI.pth', map_location='cpu'))
            parser = Parser(path = './lambda_factor/ISBI')
        else:
            train_dataset = CamSeq_dataset('./data/benli/CamSeq01_hdf5', train=True)
            pytorch_model = Segmentation_UNet(input_channel=3, class_num=32, fnum=64)
            pytorch_model.load_state_dict(torch.load('./model/benli/CamSeq01/CamSeq01.pth', map_location='cpu'))
            parser = Parser(path = './lambda_factor/CamSeq01')

        train_dataloader = DataLoader(train_dataset, batch_size=1)
        norm_data = []
        for data, _ in train_dataloader:
            norm_data.append(data)
        norm_data = np.vstack(norm_data)
        norm_data = norm_data[:64]
        norm_data_tensor = torch.from_numpy(norm_data)
        norm_data_tensor = norm_data_tensor.type(torch.FloatTensor)
        parser_model = parser.parse(pytorch_model,norm_data_tensor, method=method, scale_method=scale_method)
        snn_model = parser.convert_to_snn(parser_model, neuron_class=neuron_class, timesteps=timesteps, reset_method=reset_method, v_threshold=vth)
        snn_model.to(device)

        evaluate(snn_model, base_path=base_path, dataset_name=dataset_name, timesteps=timesteps, neuron_class=neuron_class, vth=vth, record_timestep=record,
            reset_method=reset_method, scale_method=scale_method, method=method, device=device, test=test)
