# type:ignore
import os
import cv2
import torch
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from torch import optim
from snn.conversion import Parser
from model.ann_model.U_Net import *
from snn import simulation
from snn.tools.file_tools import check_exist_makedirs
from dataset_process.DRIVE_dataset import DRIVE_dataset, load_hdf5
from dataset_process.ISBI_dataset import train_ISBI_dataset, test_ISBI_dataset
from dataset_process.CamSeq_dataset import CamSeq_dataset
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

def train_snn(snn_model, path, logs_path, model_path, dataset_name, opts, epochs, batch_size, timesteps, device, test):

    if dataset_name == 'DRIVE':
        data_path = './data/benli/DRIVE'
        train_dataset = DRIVE_dataset(data_path, train=True)
        test_dataset = DRIVE_dataset(data_path, train=False)
        border_masks = load_hdf5(os.path.join(data_path, 'DRIVE_dataset_borderMasks_test.hdf5'))
        kwargs = {'border_masks': border_masks}
    elif dataset_name == 'ISBI':
        data_path = './data/benli/ISBI_2012'
        train_dataset = train_ISBI_dataset(os.path.join(data_path, 'train'))
        test_dataset = test_ISBI_dataset(os.path.join(data_path, 'test'))
        kwargs = {}
    else:
        data_path = './data/benli/CamSeq01_hdf5'
        train_dataset = CamSeq_dataset(data_path, train=True)
        test_dataset = CamSeq_dataset(data_path, train=False)
        kwargs = {'id2code': test_dataset.id2code}

    train_dataloader = DataLoader( 
        dataset=train_dataset,
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )

    if dataset_name == 'DRIVE':
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=156,
            shuffle=False
        )
    else:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False
        )

    writer = SummaryWriter(logs_path)

    snn_model.to(device)
    
    simulator = simulation.simulator(timesteps=timesteps, dataset_name=dataset_name, 
                                     path=path, logs_path=logs_path, device=device, **kwargs)

    if opts == 'adam':
        optimizer = optim.Adam(snn_model.parameters(), lr=learning_rate)
    elif opts == 'sgd':
        optimizer = optim.SGD(snn_model.parameters(), lr=learning_rate)
    elif opts == 'rms':
        optimizer = optim.RMSprop(snn_model.parameters(), lr=learning_rate)
    elif opts == 'adadelta':
        optimizer = optim.Adaelta(snn_model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    if dataset_name in ['DRIVE', 'ISBI']:
        best_accuracy = 0.0
        best_jaccard_index = 0.0
        best_F1_score = 0.0
    else:
        best_mIoU = 0.0

    start = datetime.datetime.now()
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        if test:
            if epoch == 1:
                break

        running_loss = 0.0
        for idx, (train_data, train_label) in enumerate(train_dataloader):
            if test:
                if idx == 1:
                    break
            optimizer.zero_grad()
            train_data = train_data.to(device=device, dtype=torch.float32)
            train_label = train_label.to(device=device, dtype=torch.long)
            train_label = torch.squeeze(train_label, dim=1)

            output = simulator.simulate_for_sample(snn_model, train_data)
            loss = criterion(output, train_label)
            running_loss += loss
            # 更新参数
            loss.backward()
            optimizer.step()

        if dataset_name in ['DRIVE', 'ISBI']:
            pred_list, accuracy, jaccard_index, F1_score = simulator.simulate_mem(snn_model, 
                                                                                  test_dataloader, 
                                                                                  eval_train=True, 
                                                                                  record_timestep=False, 
                                                                                  test=test)
            if F1_score > best_F1_score:
                best_accuracy = accuracy
                best_jaccard_index = jaccard_index
                best_F1_score = F1_score
                for index in range(len(pred_list)):
                    result_path = os.path.join(path, str(index) + '.png')
                    cv2.imwrite(result_path, pred_list[index] * 255)
                    writer.add_image(str(index) + "_results", pred_list[index] * 255, dataformats='hwc') 

                with open(os.path.join(path, 'results.txt'), 'w') as f:
                    f.write('ACC is ' + str(best_accuracy) + '\n'
                            'JS is ' + str(best_jaccard_index) + '\n'
                            'F1 is ' + str(best_F1_score) + '\n')
                param_dict = OrderedDict()
                for key, value in snn_model.state_dict().items():
                    if 'weight' in key or 'bias' in key:
                        param_dict[key] = value
                        
                torch.save(param_dict, model_path)

            writer.add_scalar('evaluate acc', accuracy, epoch)
            writer.add_scalar('evaluate JS', jaccard_index, epoch)
            writer.add_scalar('evaluate F1', F1_score, epoch)
            writer.add_scalar('best acc', best_accuracy, epoch)
            writer.add_scalar('best JS', best_jaccard_index, epoch)
            writer.add_scalar('best F1', best_F1_score, epoch)
            writer.add_scalar("training loss", running_loss / len(train_dataloader) , epoch)             
        else:
            pred_list, mIoU = simulator.simulate_mem(snn_model, 
                                                     test_dataloader, 
                                                     eval_train=True, 
                                                     record_timestep=False, 
                                                     test=test)
            if mIoU > best_mIoU:
                best_mIoU = mIoU
                for index in range(len(pred_list)):
                    result_path = os.path.join(path, str(index) + '.png')
                    cv2.imwrite(result_path, pred_list[index])
                    writer.add_image(str(index) + "_results", pred_list[index], dataformats='hwc') 

                with open(os.path.join(path, 'results.txt'), 'w') as f:
                    f.write('mIoU is ' + str(best_mIoU) + '\n')
                
                param_dict = OrderedDict()
                for key, value in snn_model.state_dict().items():
                    if 'weight' in key or 'bias' in key:
                        param_dict[key] = value
                        
                torch.save(param_dict, model_path)

            writer.add_scalar('evaluate mIoU', mIoU, epoch)
            writer.add_scalar('best mIoU', best_mIoU, epoch)
            writer.add_scalar("training loss", running_loss / len(train_dataloader) , epoch)             

    end = datetime.datetime.now()
    time_mean = ((end - start) / epochs).seconds
    format_time = datetime.timedelta(seconds=time_mean)
    print("each epoch mean time: {}".format(format_time))
    writer.add_text("training", "each epoch mean time:{}".format(format_time))  
    
    if dataset_name in ['DRIVE', 'ISBI']:
        writer.add_text("testing/acc", "evaluation acc:{}".format(best_accuracy))
        writer.add_text("testing/JS", "evaluation JS:{}".format(best_jaccard_index))
        writer.add_text("testing/F1", "evaluation F1:{}".format(best_F1_score))
    else:
        writer.add_text("testing/mIoU", "evaluation mIoU:{}".format(best_mIoU))
    print("training done")
    writer.close()

def evaluate(snn_model, path, logs_path, dataset_name, timesteps, test):
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
    snn_simulator.simulate_mem(snn_model, test_dataloader, eval_train=False, record_timestep=False, test=test)

argparser = argparse.ArgumentParser()
argparser.add_argument('-n',      type=str,   dest='dataset_name',  required=True)
argparser.add_argument('-m',      type=str,   dest='method',        default='connection_wise', required=False)
argparser.add_argument('-s1',     type=int,   dest='seed1',         default=0,              required=False)
argparser.add_argument('-s2',     type=int,   dest='seed2',         default=0,              required=False)
argparser.add_argument('-s3',     type=int,   dest='seed3',         default=0,              required=False)
argparser.add_argument('-b',      type=int,   dest='batch_size',    default=8,              required=False)
argparser.add_argument('-t',      type=int,   dest='timesteps',     default=10,             required=True)
argparser.add_argument('-e',      type=int,   dest='epochs',        default=100,            required=False)
argparser.add_argument('-lr',     type=float, dest='learning_rate', default=1e-6,           required=False)
argparser.add_argument('-s',      type=str,   dest='scale_method',  default='robust',       required=False)
argparser.add_argument('-op',     type=str,   dest='optim',         default='adam',         required=False)
argparser.add_argument('-vth',    type=float, dest='vth',           default=1.0,            required=False)
argparser.add_argument('-T',      dest='train',  default=False,  action='store_true')
argparser.add_argument('-d',      dest='sub',    default=False,  action='store_true')
argparser.add_argument('-neuron', type = str,  dest = 'neuron_class', default='multi',   required = False)


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

# gpu           = args.gpu
method        = args.method
epochs        = args.epochs
timesteps     = args.timesteps
batch_size    = args.batch_size
dataset_name  = args.dataset_name
learning_rate = args.learning_rate
scale_method  = args.scale_method
train         = args.train
sub           = args.sub
opts          = args.optim
neuron_class  = args.neuron_class
vth           = args.vth


if sub:
    reset_method = 'reset_by_subtraction'
else:
    reset_method = 'reset_by_zero'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_path      = './test/seg_train'
eval_path      = './test/seg_train_eval'

str_n_join = ['optim', opts, 'batch_size', batch_size, 'lr', learning_rate, 'ep', epochs, 'm', method, 't', timesteps, 'neuron', neuron_class]
str_s_join = ['s1', seed1, 's2', seed2, 's3', seed3] + str_n_join
str_s = '_'.join(str(s) for s in str_s_join)
str_n = '_'.join(str(s) for s in str_n_join)

post_path = os.path.join(dataset_name, scale_method, reset_method, str_n)
post_log_path = os.path.join('logs', dataset_name, scale_method, reset_method, str_s)
print(post_path, post_log_path)

path           = os.path.join(base_path, post_path)
logs_path      = os.path.join(base_path, post_log_path)
model_path     = os.path.join(path, 'snn_model', 'snn_model.pth')
eval_path      = os.path.join(eval_path, post_path)
eval_logs_path = os.path.join(eval_path, post_log_path)
check_exist_makedirs(os.path.join(path, 'snn_model'))

# 用于测试代码是否跑通: 测试的时候为True
test = True

if train:
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
        parser = Parser(path='./lambda_factor/CamSeq01')
    
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

    train_snn(snn_model, path, logs_path, model_path, dataset_name, opts, epochs, batch_size, timesteps, device=device, test=test)
else:
    if dataset_name == 'DRIVE':
        pytorch_model = Segmentation_UNet(input_channel=1, class_num=2, fnum=64) 
        parser = Parser(path = './lambda_factor/DRIVE')
    elif dataset_name == 'ISBI':
        pytorch_model = Segmentation_UNet(input_channel=1, class_num=2, fnum=64) 
        parser = Parser(path = './lambda_factor/ISBI')
    else:
        pytorch_model = Segmentation_UNet(input_channel=3, class_num=32, fnum=64) 
        parser = Parser(path='./lambda_factor/CamSeq01')

    snn_model = parser.convert_to_snn(pytorch_model, neuron_class=neuron_class, timesteps=timesteps, reset_method=reset_method, v_threshold=vth)

save_dict = torch.load(model_path)
snn_model_dict = snn_model.state_dict()
load_dict_keys = [k for k in snn_model.state_dict().keys() if 'weight' in k or 'bias' in k]
state_dict = {k:v for k,v in save_dict.items() if k in load_dict_keys}
snn_model_dict.update(state_dict)
snn_model.load_state_dict(snn_model_dict)
evaluate(snn_model, eval_path, eval_logs_path, dataset_name, timesteps, test=test)

