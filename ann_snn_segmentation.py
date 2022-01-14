import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset_process.ISBI_dataset import train_ISBI_dataset, test_ISBI_dataset
from dataset_process.DRIVE_dataset import DRIVE_dataset
from dataset_process.CamSeq_dataset import CamSeq_dataset, load_hdf5, onehot_to_rgb
from snn.conversion.utils import Parser
import snn.simulation.utils as simulation
import argparse
import configparser   
from tools.file_tools import check_exist_makedirs

def model_select(input_channel, output_channel, version, network_input_channels):
    if version == 'original':
        from model.ann_model.U_Net import Segmentation_UNet
        model = Segmentation_UNet(input_channel=input_channel, class_num=output_channel, fnum=network_input_channels)
    elif version == 'v1':
        from model.ann_model.U_Net import Segmentation_UNet_shallow_v1
        model = Segmentation_UNet_shallow_v1(input_channel=input_channel, class_num=output_channel, fnum=network_input_channels)
    elif version == 'v2':
        from model.ann_model.U_Net import Segmentation_UNet_shallow_v2
        model = Segmentation_UNet_shallow_v2(input_channel=input_channel, class_num=output_channel, fnum=network_input_channels)
    elif version == 'v3':
        from model.ann_model.U_Net import Segmentation_UNet_shallow_v3
        model = Segmentation_UNet_shallow_v3(input_channel=input_channel, class_num=output_channel, fnum=network_input_channels)

    return model


def contain_any(seq, aset):
    return True if any(i in seq for i in aset) else False


def params_dict(model, agg, threshold, output_name=None):
    params_dict = {}
    if output_name is not None:
        params_dict[output_name] = threshold
    module_req = ['relu', 'pooling']
    if agg:
        count = 0 
        for name, module in model._modules.items():
            if contain_any(name, module_req):
                count += 1
        order = 0
        for name, module in model._modules.items():
            if contain_any(name, module_req):
                params_dict[name] = 0.9 + (threshold - 0.9) / count * order
                # params_dict[name] = 1.0 - (threshold - 0.9) / count * order
                order = order + 1
    else:
        for name, module in model._modules.items():
            if contain_any(name, module_req):
                params_dict[name] = threshold
    return params_dict

def part_of_train_data(train_dataloader):
    norm_data = []
    for data, label in train_dataloader:
        norm_data.append(data)
    norm_data = np.vstack(norm_data)
    norm_data = norm_data[:64]
    norm_data_tensor = torch.FloatTensor(norm_data)
    return norm_data_tensor


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g', type=str, dest='gpu', required=True)
    argparser.add_argument('-m', type=str, default='no_method', dest='method', required=False)
    argparser.add_argument('-t', type=int, dest='timesteps', required=True)
    argparser.add_argument('-v', type=str, choices=['original', 'v1', 'v2', 'v3'], dest='version', required=True)
    argparser.add_argument('--cfg', type=str, default=None, dest='config_path', required=False)
    argparser.add_argument('-th', type=float, default=1.0, dest='threshold', required=False)
    argparser.add_argument('-r', type=str, choices=['zero','sub'], dest='reset_method',required=True)
    argparser.add_argument('-nc', type=int, choices=[32, 64], dest='network_input_channels', required=True)
    argparser.add_argument('-o', type=str, choices=['mem', 'spike'], dest='output_format', required=True)
    argparser.add_argument('-c', type=str, choices=['ann', 'snn'], dest='conversion', required=True)
    argparser.add_argument('-avg', dest='avg', default=False, action='store_true')
    argparser.add_argument('-agg', dest='agg', default=False, action='store_true')
    args = argparser.parse_args()

    if args.config_path is not None:
        config_path = args.config_path

    gpu = args.gpu
    method = args.method
    version = args.version 
    timesteps = args.timesteps
    threshold = args.threshold
    reset = args.reset_method
    output_format = args.output_format
    conversion = args.conversion
    agg = args.agg
    avg = args.avg

    if reset == 'zero':
        reset_method = 'reset_by_zero'
    elif reset == 'sub':
        reset_method = 'reset_by_subtraction'
    network_input_channels = args.network_input_channels

    # color = 'RGB'
    color = 'GRAY'
    task = 'segmentation'
    batch_size = 1
    # dataset_name = 'CamSeq01'
    # dataset_name = "ISBI"
    dataset_name = "DRIVE"
    # data_path = './data/CamSeq01_hdf5'
    # data_path = './data/ISBI_2012'
    data_path = './data/DRIVE'
    base_path = os.path.join('./output2', 'snn', dataset_name)
    if conversion == 'ann':
        if agg:
            if avg:
                save_path = os.path.join(base_path, reset_method, 
                    method + '_' + version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_steps' + str(timesteps) + '_eval_agg_avg',
                )
                log_dir = os.path.join('./output2', 'logs', 'snn', 
                    dataset_name, 
                    reset_method,
                    method + '_' + version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_steps' + str(timesteps) + '_eval_agg_avg'
                )
            else:
                save_path = os.path.join(base_path, reset_method, 
                    method + '_' + version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_steps' + str(timesteps) + '_eval_agg',
                )
                log_dir = os.path.join('./output2', 'logs', 'snn', 
                    dataset_name, 
                    reset_method,
                    method + '_' + version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_steps' + str(timesteps) + '_eval_agg'
                )
            lambda_path = os.path.join('./lambda_factor', dataset_name, version, str(network_input_channels))
            model_path = os.path.join('./model/ann_model', dataset_name, str(network_input_channels), version + '.pth')
        else:
            if avg:
                save_path = os.path.join(base_path, reset_method, 
                    method + '_' + version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_steps' + str(timesteps) + '_eval_avg'
                )
                log_dir = os.path.join('./output2', 'logs', 'snn', 
                    dataset_name, 
                    reset_method,
                    method + '_' + version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_steps' + str(timesteps) + '_eval_avg'
                )
            else:
                save_path = os.path.join(base_path, reset_method, 
                    method + '_' + version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_steps' + str(timesteps) + '_eval'
                )
                log_dir = os.path.join('./output2', 'logs', 'snn', 
                    dataset_name, 
                    reset_method,
                    method + '_' + version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_steps' + str(timesteps) + '_eval'
                )
            lambda_path = os.path.join('./lambda_factor', dataset_name, version, str(network_input_channels))
            model_path = os.path.join('./model/ann_model', dataset_name, str(network_input_channels), version + '.pth')
    else:
        save_path = os.path.join(base_path, reset_method, 
            version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_steps' + str(timesteps) + '_train'
        )
        log_dir = os.path.join('./output2', 'logs', 'snn', 
            dataset_name, 
            reset_method,
            version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold)
        )
        model_path = os.path.join('./model/snn_model', dataset_name, str(network_input_channels), output_format + '_' + version + '.pth')


    config_path = os.path.join(save_path, 'config.txt')
    # lambda_path = os.path.join('./lambda_factor', 'CamSeq', version, str(network_input_channels))
    # model_path = os.path.join('./model/ann_model', dataset_name, version + '.pth')

    print(save_path, log_dir)
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

    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')

    if conversion == 'ann':
        if dataset_name == 'CamSeq01':
            train_dataset = CamSeq_dataset(data_path, train=True)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        elif dataset_name == 'ISBI':
            train_datapath = os.path.join(data_path, 'train')
            train_dataset = train_ISBI_dataset(train_datapath)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        elif dataset_name == 'DRIVE':
            train_dataset = DRIVE_dataset(data_path, train=True)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        norm_data_tensor = part_of_train_data(train_dataloader)

        # model
        model = model_select(input_channel, output_channel, version, network_input_channels)
        # model.load_state_dict(torch.load(model_path, map_location={'cuda:1':'cuda:0'})) 
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        
        parser = Parser(path=lambda_path,
                        task='segmentation',
                        device=device)

        if method != 'no_method':
            parser_model = parser.parse(model, norm_data_tensor, method=method)
            
    # parser = Parser(path=lambda_path,
    #                 task='segmentation',
    #                 device=device)

    if output_format == 'spike':
        threshold_dict = params_dict(model, agg, threshold, parser.output_name)
    else:
        threshold_dict = params_dict(model, agg, threshold)

    if avg:
        if method == 'channel_wise':
            input_path = os.path.join(lambda_path, "lambda_input_channel_wise.npy")
            output_path = os.path.join(lambda_path, "lambda_output_channel_wise.npy")
            lambda_input = np.load(input_path, allow_pickle=True).item()
            lambda_output = np.load(output_path, allow_pickle=True).item()
            for key, output in lambda_output.items():
                if 'pooling' in key:
                    scale = output / lambda_input[key]
                    threshold_dict[key] = scale

    # snn_model = parser.convert_to_snn(parser_model)
    if method == 'no_method':
        snn_model = parser.convert_to_snn_specific(model, threshold_dict, reset_method, output_format, optimizer_method='no_op_vth', reduce_error='False', avg=avg)
    else:
        snn_model = parser.convert_to_snn_specific(parser_model, threshold_dict, reset_method, output_format, optimizer_method='no_op_vth', reduce_error='False', avg=avg)
    
    if dataset_name == 'CamSeq01':
        test_dataset = CamSeq_dataset(data_path, train=False)
        id2code = test_dataset.id2code
        kwargs = {'id2code': id2code}
    elif dataset_name == 'ISBI':
        test_datapath = os.path.join(data_path, 'test')
        test_dataset = test_ISBI_dataset(test_datapath)
        kwargs = {}
    elif dataset_name == 'DRIVE':
        test_border_masks = load_hdf5(os.path.join(data_path, 'DRIVE_dataset_borderMasks_test.hdf5'))
        test_dataset = DRIVE_dataset(data_path, train=False)
        kwargs = {'test_border_masks': test_border_masks}
        
    #test_dataloader = DataLoader(test_dataset, batch_size=3)
    if dataset_name == 'DRIVE':
        test_dataloader = DataLoader(test_dataset, batch_size=156)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


    snn_simulator = simulation.simulator(timesteps=timesteps,
                                        dataset=dataset_name,
                                        train=False,
                                        save_path=save_path,
                                        log_dir=log_dir,
                                        task=task,
                                        device=device, **kwargs)
    snn_simulator.simulate_spike(snn_model, test_dataset.total_num / batch_size, test_dataloader, record_timesteps_results=True, **kwargs)

    # config write
    config = configparser.ConfigParser()
    config['path'] = {
        'data_path': data_path,
        'model_path': model_path
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
        'reset_method': reset_method,
        'method': method,
        'timesteps': timesteps
    }

    config['threshold'] = threshold_dict
    with open(config_path, 'w') as configfile:
        config.write(configfile)

