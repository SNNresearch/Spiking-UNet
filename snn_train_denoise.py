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
from collections import OrderedDict

from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset_process.CBSD_dataset import train_CBSD_dataset, test_CBSD_dataset, load_hdf5
from dataset_process.Set_dataset import Set12_Dataset_hdf5
from torch.utils.tensorboard.writer import SummaryWriter

def train_snn(snn_model, path, logs_path, model_path, dataset_name, opts, epochs, batch_size, timesteps, device, test):
    if dataset_name == 'CBSD':
        data_path = './data/benli/CBSD_dataset'
        train_dataset = train_CBSD_dataset(data_path, color=True, valid=False, transform=transforms.ToTensor())
        test_dataset_25 = test_CBSD_dataset(data_path, 'CBSD_patch_test_img_sigma_' + "25.hdf5",
                                            color=True, transform=transforms.ToTensor())
    else:
        data_path = './data/benli/BSD_dataset'
        train_dataset = train_CBSD_dataset(data_path, color=False, valid=False, transform=transforms.ToTensor())
        test_dataset_25 = test_CBSD_dataset(data_path, 'BSD_patch_test_img_sigma_' + "25.hdf5",
                                            color=False, transform=transforms.ToTensor())

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test_dataloader_25 = DataLoader(
        test_dataset_25,
        batch_size=4,
        shuffle=False
    )

    best_psnr_25 = 0.0
    best_ssim_25 = 0.0

    writer = SummaryWriter(logs_path)

    snn_model.to(device)

    simulator = simulation.simulator(timesteps=timesteps, dataset_name=dataset_name,
                                     path=path, logs_path=logs_path, device=device)

    if opts == 'adam':
        optimizer = optim.Adam(snn_model.parameters(), lr=learning_rate)
    elif opts == 'sgd':
        optimizer = optim.SGD(snn_model.parameters(), lr=learning_rate)
    elif opts == 'rms':
        optimizer = optim.RMSprop(snn_model.parameters(), lr=learning_rate)
    elif opts == 'adadelta':
        optimizer = optim.Adaelta(snn_model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    criterion_mem = nn.MSELoss()

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
            train_data = train_data.to(device)
            train_label = train_label.to(device)

            output = simulator.simulate_for_sample(snn_model, train_data)
            loss = criterion(output, train_label)

            running_loss += loss
            # 更新参数
            loss.backward()
            optimizer.step()

        pred_list, avg_psnr_25, avg_ssim_25 = simulator.simulate_mem(snn_model,
                                                                     test_dataloader_25,
                                                                     eval_train=True,
                                                                     record_timestep=False,
                                                                     test=test)

        if avg_psnr_25 > best_psnr_25:
            best_psnr_25 = avg_psnr_25
            best_ssim_25 = avg_ssim_25
            for index in range(len(pred_list)):
                result_path = os.path.join(path, str(index) + '.png')
                cv2.imwrite(result_path, pred_list[index][0])
                writer.add_image(str(index) + "_results", pred_list[index][0], dataformats='hwc')

            with open(os.path.join(path, 'results.txt'), 'w') as f:
                f.write("PSNR of imgs is:" + str(best_psnr_25) + '\n')
                f.write("SSIM of imgs is:" + str(best_ssim_25) + '\n')
            param_dict = OrderedDict()
            for key, value in snn_model.state_dict().items():
                if 'weight' in key or 'bias' in key:
                    param_dict[key] = value
            torch.save(param_dict, model_path)

        writer.add_scalar('evaluate psnr', avg_psnr_25, epoch)
        writer.add_scalar('evaluate ssim', avg_ssim_25, epoch)
        writer.add_scalar('best psnr', best_psnr_25, epoch)
        writer.add_scalar('best ssim', best_ssim_25, epoch)
        writer.add_scalar("training loss", running_loss / len(train_dataloader) , epoch)

    end = datetime.datetime.now()
    time_mean = ((end - start) / epochs).seconds
    format_time = datetime.timedelta(seconds=time_mean)
    print("each epoch mean time: {}".format(format_time))
    writer.add_text("training", "each epoch mean time:{}".format(format_time))
    writer.add_text("testing/psnr_25", "evaluation psnr:{}".format(best_psnr_25))
    writer.add_text("testing/ssim_25", "evaluation ssim:{}".format(best_ssim_25))
    print("training done")
    writer.close()

def evaluate(snn_model, path, logs_path, dataset_name, timesteps, noise_level, test):
    if dataset_name == "CBSD":
        test_dataset = test_CBSD_dataset(dir_path='./data/benli/CBSD_dataset', file_name='CBSD_patch_test_img_sigma_'+ str(noise_level) + '.hdf5', color=True, transform=transforms.ToTensor())
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    elif dataset_name == "BSD":
        test_dataset = test_CBSD_dataset(dir_path='/data/benli/BSD_dataset', file_name='BSD_patch_test_img_sigma_'+ str(noise_level) + '.hdf5', color=False, transform=transforms.ToTensor())
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    else:
        test_dataset = Set12_Dataset_hdf5(data_path='/data/benli/Set12/test', noise_level=noise_level)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    snn_simulator = simulation.simulator(timesteps=timesteps, path=path, logs_path=logs_path, device=device, dataset_name=dataset_name)
    snn_simulator.simulate_mem(snn_model, test_dataloader, eval_train=False, record_timestep=False, test=test)

argparser = argparse.ArgumentParser()
argparser.add_argument('-g',    type=str,   dest='gpu',           default='0')
argparser.add_argument('-n',    type=str,   dest='dataset_name',  required=True)
argparser.add_argument('-m',    type=str,   dest='method',        default='connection_wise', required=False)
argparser.add_argument('-s1',   type=int,   dest='seed1',         default=0,              required=False)
argparser.add_argument('-s2',   type=int,   dest='seed2',         default=0,              required=False)
argparser.add_argument('-s3',   type=int,   dest='seed3',         default=0,              required=False)
argparser.add_argument('-b',    type=int,   dest='batch_size',    default=8,              required=False)
argparser.add_argument('-t',    type=int,   dest='timesteps',     default=10,             required=True)
argparser.add_argument('-e',    type=int,   dest='epochs',        default=100,            required=False)
argparser.add_argument('-lr',   type=float, dest='learning_rate', default=1e-6,           required=False)
argparser.add_argument('-s',    type=str,   dest='scale_method',  default='robust',       required=False)
argparser.add_argument('-op',   type=str,   dest='optim',         default='adam',         required=False)
argparser.add_argument('-T',      dest='train',  default=False,  action='store_true')
argparser.add_argument('-d',      dest='sub',    default=False,  action='store_true')
argparser.add_argument('-neuron', type = str,  dest = 'neuron_class', default  = 'binary',   required = False)
argparser.add_argument('-vth',    type=float, dest='vth',           default=1.0,            required=False)

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

gpu           = args.gpu
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

device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')

base_path      = './model/benli/CBSD/denoise_train'
eval_path      = './test/denoise_train_eval'

str_n_join = ['optim', opts, 'batch_size', batch_size, 'lr', learning_rate, 'ep', epochs, 'm', method, 't', timesteps, 'neuron', neuron_class]

str_s_join = ['s1', seed1, 's2', seed2, 's3', seed3] + str_n_join
str_s = '_'.join(str(s) for s in str_s_join)
str_n = '_'.join(str(s) for s in str_n_join)

post_path = os.path.join(dataset_name, scale_method, reset_method, str_n)
post_log_path = os.path.join('logs', dataset_name, scale_method, reset_method, str_s)


path           = os.path.join(base_path, post_path)
logs_path      = os.path.join(base_path, post_log_path)
model_path     = os.path.join(path, 'snn_model', 'snn_model.pth')
eval_path      = os.path.join(eval_path, post_path)
eval_logs_path = os.path.join(eval_path, post_log_path)
check_exist_makedirs(os.path.join(path, 'snn_model'))

# 用于测试代码是否跑通: 测试的时候为True
test = True
if train:
    if dataset_name == 'CBSD':
        train_data = load_hdf5(os.path.join('./data/benli/CBSD_dataset/CBSD_patch_diff_train.hdf5'))
        pytorch_model = Denoising_Model(color=True)
        pytorch_model.load_state_dict(torch.load('./model/benli/CBSD/CBSD.pth'))
        parser = Parser(path = './lambda_factor/CBSD')
    else:
        train_data = load_hdf5(os.path.join('./data/benli/BSD_dataset/BSD_patch_diff_train.hdf5'))
        pytorch_model = Denoising_Model(color=False)
        pytorch_model.load_state_dict(torch.load('./model/benli/BSD/BSD.pth'))
        parser = Parser(path='./lambda_factor/BSD')

    L=[random.randint(0,train_data.shape[0] - 1) for _ in range(32)]
    norm_data = train_data[L,:,:,:]
    norm_data = np.transpose(norm_data, (0,3,1,2))
    norm_data_tensor = torch.from_numpy(norm_data)
    norm_data_tensor = norm_data_tensor.type(torch.FloatTensor)
    parser_model = parser.parse(pytorch_model, norm_data_tensor, method=method, scale_method=scale_method)
    snn_model = parser.convert_to_snn(pytorch_model, neuron_class=neuron_class, stbp=stbp, timesteps=timesteps, reset_method=reset_method, v_threshold=vth)
    del norm_data, norm_data_tensor

    train_snn(snn_model, path, logs_path, model_path, dataset_name, opts, epochs, batch_size, timesteps, membrane=membrane, ratio=ratio, relu=relu, stbp=stbp, device=device, test=test)
else:
    if dataset_name == 'CBSD':
        pytorch_model = Denoising_Model(color=True)
        parser = Parser(path='./lambda_factor/CBSD')
    else:
        pytorch_model = Denoising_Model(color=False)
        parser = Parser(path='./lambda_factor/BSD')

    snn_model = parser.convert_to_snn(pytorch_model, neuron_class=neuron_class, stbp=stbp, timesteps=timesteps, reset_method=reset_method, v_threshold=vth)

save_dict = torch.load(model_path)
snn_model_dict = snn_model.state_dict()
load_dict_keys = [k for k in snn_model.state_dict().keys() if 'weight' in k or 'bias' in k]
state_dict = {k:v for k,v in save_dict.items() if k in load_dict_keys}
snn_model_dict.update(state_dict)
snn_model.load_state_dict(snn_model_dict)

path_15 = os.path.join(eval_path, 'noise_level_15')
path_25 = os.path.join(eval_path, 'noise_level_25')
path_35 = os.path.join(eval_path, 'noise_level_35')
logs_path_15 = os.path.join(eval_logs_path, 'noise_level_15')
logs_path_25 = os.path.join(eval_logs_path, 'noise_level_25')
logs_path_35 = os.path.join(eval_logs_path, 'noise_level_35')
evaluate(snn_model, path_15, logs_path_15, dataset_name, timesteps, 15, test=test)
evaluate(snn_model, path_25, logs_path_25, dataset_name, timesteps, 25, test=test)
evaluate(snn_model, path_35, logs_path_35, dataset_name, timesteps, 35, test=test)
