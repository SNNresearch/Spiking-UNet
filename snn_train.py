import os
import torch
import argparse
import torch.nn as nn
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


# count = 0
# order = 0
# def params_dict_func(modules, agg, threshold, output_name=None):
#     global count
#     params_dict = {}
#     for n, module in modules.named_children():
#         if len(list(module.children())) > 0:
#             params_dict = params_dict_func(module, agg, threshold, output_name)
#         else:
#             if output_name is not None:
#                 params_dict[output_name] = threshold

#             if agg:
#                 if isinstance(module, nn.ReLU):
#                     count += 1
#                 if isinstance(module, nn.AvgPool2d):
#                     count += 1
#             else:
#                 if isinstance(module, nn.ReLU):
#                     params_dict[n] = threshold
#                 if isinstance(module, nn.AvgPool2d):
#                     params_dict[n] = threshold

#     return params_dict
            
# def agg_params_dict_func(modules, agg, threshold, output_name=None):
#     global order
#     global count
#     params_dict = {}
#     for n, module in modules.named_children():
#         if len(list(module.children())) > 0:
#             params_dict = params_dict_func(module, agg, threshold, output_name)
#         else:
#             if output_name is not None:
#                 params_dict[output_name] = threshold

#             if agg:
#                 if isinstance(module, nn.ReLU):
#                     params_dict[name] = 0.9 + (threshold - 0.9) / count * order
#                     order = order + 1
#                 if isinstance(module, nn.AvgPool2d):
#                     params_dict[name] = 0.9 + (threshold - 0.9) / count * order
#                     order = order + 1

#     return params_dict

def params_dict_func(model, agg, threshold, output_name=None):
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
    argparser.add_argument('-s1', type=int, dest='seed1', required=True)
    argparser.add_argument('-s2', type=int, dest='seed2', required=True)
    argparser.add_argument('-name', type=str, dest='name', required=True)
    # argparser.add_argument('-s3', type=int, dest='seed3', required=True)
#     argparser.add_argument('-g', type=str, dest='gpu', required=True)
    argparser.add_argument('-b', type=int, dest='batch_size', required=True)
    argparser.add_argument('-m', type=str, choices=['no_method', 'layer_wise', 'channel_wise', 'group_wise'], dest='method', required=True)
    argparser.add_argument('-t', type=int, dest='timesteps', required=True)
    argparser.add_argument('-v', type=str, choices=['original', 'v1', 'v2', 'v3'], dest='version', required=True)
    argparser.add_argument('--cfg', type=str, default=None, dest='config_path', required=False)
    argparser.add_argument('-th', type=float, default=1.0, dest='threshold', required=False)
    argparser.add_argument('-r', type=str, choices=['zero','sub'], dest='reset_method',required=True)
    argparser.add_argument('-nc', type=int, choices=[32, 64], dest='network_input_channels', required=True)
    argparser.add_argument('-o', type=str, choices=['mem', 'spike'], dest='output_format', required=True)
    argparser.add_argument('-lr', type=float, dest='learning_rate', required=True)
    argparser.add_argument('-th_lr', type=float, default=0.0001, dest='th_learning_rate')
    argparser.add_argument('-op', type=str, choices=['adam', 'RMS'], dest='optimizer_class', required=True)
    argparser.add_argument('-e', type=int, dest='epochs', required=True)
    # argparser.add_argument('-op_vth', type=str, choices=['no_op', 'op', 'only_op'], dest='optimizer_vth', required=True)
    argparser.add_argument('-op_method', type=str, choices=['no_op_scale', 'no_op_vth', 'op_vth', 'only_op_vth', 'op_eth', 'only_op_eth', 'stbp'], dest='optimizer_method', required=True)
    argparser.add_argument('-re', dest='reduce_error', default=False, action='store_true')
    argparser.add_argument('-agg', dest='agg', default=False, action='store_true')
    argparser.add_argument('-avg', dest='avg', default=False, action='store_true')
    # argparser.add_argument('-bias', dest='bias', default=False, action='store_true')
    argparser.add_argument('-tm', type=str, dest='train_method', required=True)

    args = argparser.parse_args()

    if args.config_path is not None:
        config_path = args.config_path

#     gpu = args.gpu
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

    batch_size = args.batch_size
    dataset_name = args.name
    method = args.method
    train_method = args.train_method
    version = args.version 
    timesteps = args.timesteps
    threshold = args.threshold
    reset = args.reset_method
    output_format = args.output_format
    learning_rate = args.learning_rate
    th_learning_rate = args.th_learning_rate
    optimizer_class = args.optimizer_class
    optimizer_method = args.optimizer_method
    epochs = args.epochs
    reduce_error = args.reduce_error
    agg = args.agg
    avg = args.avg
    # bias = args.bias

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
        elif dataset_name == 'DRIVE':
            data_path = '/data/benli/DRIVE_dataset'
    task = 'segmentation'
    # dataset_name = 'CamSeq01'
    # dataset_name = 'DRIVE'
    # dataset_name = 'ISBI'
    # data_path = './data/ISBI_2012'
    # data_path = './data/CamSeq01_hdf5'
    # data_path = './data/DRIVE'
    base_path = os.path.join('/output', 'snn', dataset_name)
    if optimizer_method == 'op_vth' or optimizer_method == 'op_eth':
        train_method = train_method + '_' + str(th_learning_rate)
    save_path = os.path.join(base_path, reset_method, 
        method + '_' + version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_steps' + str(timesteps) + '_train',
        's1_'+ str(seed1) + '_s2_' + str(seed2) + '_batch_size' + str(batch_size) + '_lr' + str(learning_rate) + '_ep' + str(epochs) + '_' + optimizer_class + '_' + train_method
    )
    config_path = os.path.join(save_path, 'config.txt')
    log_dir = os.path.join('/output', 'logs', 'snn', 
        dataset_name, 
        reset_method,
        method + '_' + version + '_' + 'in' + str(network_input_channels) + '_' + output_format + '_' + 'th' + str(threshold) + '_train',
        's1_'+ str(seed1) + '_s2_' + str(seed2) + '_batch_size' + str(batch_size) + '_steps' + str(timesteps) + '_lr' + str(learning_rate) + '_ep' + str(epochs) + '_' + optimizer_class + '_' + train_method
    )
    lambda_path = os.path.join('/code/lambda_factor', dataset_name, version, str(network_input_channels))
    model_path = os.path.join('/model/benli/unet_zoo', dataset_name, version + '.pth')
    model_save = os.path.join('/model/benli/unet_zoo/snn', dataset_name, str(network_input_channels))
    model_save_path = os.path.join(model_save, 's1_'+ str(seed1) + '_s2_' + str(seed2) + '_batch_size' + str(batch_size) + '_' + method + '_steps' + str(timesteps) + '_' + train_method + '_' + version + '.pth')

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
    
    if optimizer_method == 'stbp':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
#         device = torch.device(('cuda:' + gpu) if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    if dataset_name == 'DRIVE':
        # current_device = str(torch.cuda.current_device())
        model.load_state_dict(torch.load(model_path, map_location={'cuda:2':'cuda:0'})) 
    else:
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    parser = Parser(
        path=lambda_path,
        task='segmentation',
        device=device
    )

    if method != 'no_method':
        parser_model = parser.parse(model, norm_data_tensor, method=method)

    if output_format == 'spike':
        threshold_dict = params_dict_func(model, agg, threshold, parser.output_name)
    else:
        threshold_dict = params_dict_func(model, agg, threshold)

    print(avg)
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
        elif method == 'layer_wise':
            input_path = os.path.join(lambda_path, "lambda_input_channel_wise.npy")
            output_path = os.path.join(lambda_path, "lambda_output_channel_wise.npy")
            lambda_input = np.load(input_path, allow_pickle=True).item()
            lambda_output = np.load(output_path, allow_pickle=True).item()
            for key, output in lambda_output.items():
                if 'pooling' in key:
                    scale = output / lambda_input[key]
                    threshold_dict[key] = scale
        # print(threshold_dict)

    if method != 'no_method':
        if method == 'channel_wise':
            snn_model = parser.convert_to_snn_specific(parser_model, threshold_dict, reset_method, output_format, optimizer_method, reduce_error, avg, timesteps=timesteps)
        elif method == 'layer_wise':
            snn_model = parser.convert_to_snn_specific(parser_model, threshold_dict, reset_method, output_format, optimizer_method, reduce_error, avg=False, timesteps=timesteps)
    else:
        snn_model = parser.convert_to_snn_specific(model, threshold_dict, reset_method, output_format, optimizer_method, reduce_error, avg)


    # snn_model = parser.convert_to_snn(parser_model)
    if optimizer_method == 'stbp':
        snn_model = parallel_v2(snn_model)
    else:
        snn_model.to(device)

    # print(snn_model)

    # snn_model = torch.nn.DataParallel(snn_model)

    vth_list = []
    down_vth_list = []
    up_vth_list = []
    weights_list = []
    # if optimizer_method == 'op_vth' or optimizer_method == 'op_eth':
    #     flag = 0
    #     for name, param in snn_model.named_parameters():
    #         if 'v_threshold' in name:
    #             # vth_list.append(param)
    #             if 'convtranspose' in name:
    #                 flag = 1
    #             if flag == 0:
    #                 down_vth_list.append(param)
    #             else:
    #                 up_vth_list.append(param)
    #         else:
    #             weights_list.append(param)

    #     if optimizer_class == 'adam':
    #         # optimizer = optim.Adam([{'params': weights_list}, {'params': vth_list}], lr=learning_rate)
    #         optimizer = optim.Adam([{'params': weights_list}, {'params': vth_list}], lr=learning_rate)
    #     elif optimizer_class == 'RMS':
    #         # optimizer = optim.RMSprop([{'params': weights_list}, {'params': vth_list}], lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    #         optimizer = optim.RMSprop([{'params': weights_list}, {'params': down_vth_list}, {'params': up_vth_list}], lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    #     optimizer.param_groups[0]['lr'] = learning_rate
    #     optimizer.param_groups[1]['lr'] = th_learning_rate
    #     optimizer.param_groups[2]['lr'] = 0.0001
    if optimizer_method == 'op_vth' or optimizer_method == 'op_eth':
        for name, param in snn_model.named_parameters():
            if 'v_threshold' in name:
                vth_list.append(param)
            else:
                weights_list.append(param)
        
        if optimizer_class == 'adam':
            optimizer = optim.Adam([{'params': weights_list}, {'params': vth_list}], lr=learning_rate)
        elif optimizer_class == 'RMS':
            optimizer = optim.RMSprop([{'params': weights_list}, {'params': vth_list}], lr=learning_rate, weight_decay=1e-8, momentum=0.9)
        optimizer.param_groups[0]['lr'] = learning_rate
        optimizer.param_groups[1]['lr'] = th_learning_rate


    elif optimizer_method == 'only_op_vth' or optimizer_method == 'only_op_eth':
        for name, param in snn_model.named_parameters():
            if 'v_threshold' not in name:
                param.requires_grad = False

        if optimizer_class == 'adam':
            optimizer = optim.Adam(snn_model.parameters(), lr=learning_rate)
        elif optimizer_class == 'RMS':
            optimizer = optim.RMSprop(snn_model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    else:
        # if not bias:
        #     for name, param in snn_model.named_parameters():
        #         if 'bias' in name:
        #             param.requires_grad = False
        if optimizer_class == 'adam':
            optimizer = optim.Adam(snn_model.parameters(), lr=learning_rate)
        elif optimizer_class == 'RMS':
            optimizer = optim.RMSprop(snn_model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)

    train = True
    if optimizer_method == 'stbp':
        from snn.stbp_train import train_and_evaluate
        train_and_evaluate(
            snn_model, device, optimizer_method,
            train, data_path, log_dir, 
            model_save_path, save_path, 
            dataset_name, timesteps, output_format, 
            optimizer, epochs=epochs, batch_size=batch_size)
    else:
        from snn.asf_bp_train import train_and_evaluate
        train_and_evaluate(
            snn_model, device, optimizer_method,
            train, data_path, log_dir, 
            model_save_path, save_path, 
            dataset_name, timesteps, output_format, 
            optimizer, epochs=epochs, batch_size=batch_size)

    # config write
    config = configparser.ConfigParser()
    config['path'] = {
        'data_path': data_path,
        'lambda_path': lambda_path,
        'model_path': model_path,
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
        'method': method,
        'timesteps': timesteps,
        'optimizer': optimizer_class,
        'learning_rate': learning_rate
    }

    config['threshold'] = threshold_dict
    with open(config_path, 'w') as configfile:
        config.write(configfile)
