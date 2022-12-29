# type: ignore
import os
import cv2
import torch
# import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..tools import denoising as denoise
from ..tools import segmentation as seg
from ..tools import compute_flops as cf
from ..tools.file_tools import check_exist_makedirs
from ..tools.data_clean import outliers_proc
from ..tools.segmentation.extract_patches import kill_border, recompone

from torch.utils.tensorboard import SummaryWriter

class simulator(object):
    def __init__(self, timesteps=100, dataset_name=None, path=None, logs_path=None, device='cpu', **kwargs):
        self.device = device
        self.timesteps = timesteps
        self.path = path
        self.logs_path = logs_path
        self.dataset_name = dataset_name
        check_exist_makedirs(self.path)
        check_exist_makedirs(self.logs_path)

        if self.dataset_name == 'CamSeq01':
            self.id2code = kwargs['id2code']
        elif self.dataset_name == 'DRIVE':
            self.border_masks = kwargs['border_masks']

        if os.path.exists(os.path.join(self.path, 'results.txt')):
            os.remove(os.path.join(self.path, 'results.txt'))

        if os.path.exists(os.path.join(self.path, 'result_every_img.txt')):
            os.remove(os.path.join(self.path, 'result_every_img.txt'))    

    def simulate_for_sample(self, model, data):
        model = model.eval()
        self.reset_snn(model)
        self.set_state(model, 'eval')
        with torch.no_grad():
            static_coding = model.conv1(data)
            for time in range(1, self.timesteps + 1):
                if time == 1:
                    output_mem = model.forward(static_coding, 'img')
                else:
                    output_mem += model.forward(static_coding, 'img')

        model.train()
        self.set_state(model, 'train')
        with torch.enable_grad():
            static_coding = model.conv1(data * self.timesteps)
            output = model.forward(static_coding, 'img')
            output = output / self.timesteps

        return output

    def simulate_mem(self, model, dataloader, eval_train=False, record_timestep=False, test=False):

        if self.dataset_name in ['Set12','BSD', 'CBSD']:
            psnr_list = []
            ssim_list = []
            dict_store_psnr = {}
            dict_store_ssim = {}
            dict_flops_assumption = {}
        else:
            if self.dataset_name == 'CamSeq01':
                from dataset_process.CamSeq_dataset import onehot_to_rgb
                IOU = seg.IOUMetric(num_classes=32)

            pred_list = []
            dict_store_results = {}
            dict_flops_assumption = {}
            store_labels = []

        if eval_train:
            pred_list = []
        else:
            flops = []
            writer = SummaryWriter(self.logs_path)

        model = model.eval()
        model.to(self.device)
        
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
                if test:
                    if batch_idx == 1:
                        break

                self.set_state(model, 'eval')
                self.reset_snn(model)

                if self.dataset_name in ['Set12','BSD', 'CBSD']:
                    data = data.to(self.device, dtype=torch.float32)
                    label = label.permute(0,2,3,1).cpu().numpy()
                    if self.dataset_name != 'Set12':
                        label = label * 255
                else:
                    data = data.to(self.device, dtype=torch.float32)
                    label = label.cpu().numpy()

                    if self.dataset_name == 'DRIVE':
                        label = np.expand_dims(label, axis=3)
                        label = recompone(label, 13, 12)
                        border_mask = self.border_masks[range(batch_idx, batch_idx+1)]
                        kill_border(label, border_mask)
                        label = np.squeeze(label, axis=0)

                    store_labels.append(label)
                        
                static_coding = model.conv1(data)
                if batch_idx == 0 and not eval_train:
                    batch_size = data.shape[0]
                    inp = data[range(0,1),:,:,:]
                    out = static_coding[range(0,1),:,:,:]
                    flops_first = cf.compute_flops(model.conv1, inp, out)
                    
                for time in range(1, self.timesteps + 1):
                    if time == 1:
                        output_mem = model.forward(static_coding, 'img')
                    else:
                        output_mem += model.forward(static_coding, 'img')
                    
                    if record_timestep:
                        if time % 10 == 0:
                            str_time = str(time)
                            acc_output = 0
                            for name, module in model.named_children():
                                if 'relu' in name:
                                    acc_output += torch.sum(module.spikecount)
                            try:
                                dict_flops_assumption[str_time].append(acc_output.cpu().numpy())
                            except:
                                dict_flops_assumption[str_time] = [acc_output.cpu().numpy()] 

                            store_mem = output_mem

                            if self.dataset_name in ['Set12', 'BSD', 'CBSD']:
                                store_mem = store_mem / time
                                store_mem = store_mem.permute(0,2,3,1).cpu().numpy()
                                if self.dataset_name == 'Set12':
                                    _, time_psnr, time_ssim = denoise.image_save_psnr(pic_path=os.path.join(self.path, str_time), 
                                                                                    orignal=data,
                                                                                    noise=store_mem,
                                                                                    groundtruth=label, 
                                                                                    batch_idx=batch_idx,
                                                                                    save_flag=False)
                                else:
                                    _, time_psnr, time_ssim = denoise.merge_and_psnr(pic_path=os.path.join(self.path, str_time),
                                                                                    orignal=data,
                                                                                    noise=store_mem,
                                                                                    groundtruth=label,
                                                                                    patches=4,
                                                                                    batch_idx=batch_idx,
                                                                                    poisson=False,
                                                                                    save_flag=False)
                                try:
                                    dict_store_psnr[str_time].append(time_psnr)
                                    dict_store_ssim[str_time].append(time_ssim)
                                except:
                                    dict_store_psnr[str_time] = [time_psnr]
                                    dict_store_ssim[str_time] = [time_ssim]
                            else:
                                store_mem = store_mem.cpu().numpy()
                                results = np.argmax(store_mem, axis=1)
                                if self.dataset_name == 'DRIVE':
                                    results = np.expand_dims(results, axis=3)
                                    results = recompone(results, 13, 12)
                                    results = np.squeeze(results, axis=0)
                                try:
                                    dict_store_results[str_time].append(results)
                                except:
                                    dict_store_results[str_time] = [results]
                
                if self.dataset_name in ['CBSD', 'BSD', 'Set12']:
                    output_mem = output_mem / self.timesteps
                    output_mem = output_mem.permute(0,2,3,1).cpu().numpy()
                       
                    if self.dataset_name in ['CBSD', 'BSD']:
                        if eval_train:
                            prediction, psnr, ssim = denoise.merge_and_psnr(pic_path=self.path,
                                                                            orignal=data,
                                                                            noise=output_mem,
                                                                            groundtruth=label,
                                                                            patches=4,
                                                                            batch_idx=batch_idx,
                                                                            poisson=False,
                                                                            save_flag=False)
                            pred_list.append(prediction)
                        else:
                            prediction, psnr, ssim = denoise.merge_and_psnr(pic_path=self.path,
                                                                            orignal=data,
                                                                            noise=output_mem,
                                                                            groundtruth=label,
                                                                            patches=4,
                                                                            batch_idx=batch_idx,
                                                                            poisson=False,
                                                                            save_flag=True)
                            with open(os.path.join(self.path, 'result_every_img.txt'), 'a+') as f:
                                f.write(str(batch_idx) + '_img: PSNR:' + str(psnr) + ' SSIM:' + str(ssim) + '\n')
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                    elif self.dataset_name == 'Set12':
                        prediction, psnr, ssim = denoise.image_save_psnr(pic_path=self.path,
                                                                        orignal=data, 
                                                                        noise=output_mem,
                                                                        groundtruth = label,
                                                                        batch_idx=batch_idx,
                                                                        save_flag=True)
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        with open(os.path.join(self.path, 'result_every_img.txt'), 'a+') as f:
                            f.write(str(batch_idx) + '_img: PSNR:' + str(psnr) + ' SSIM:' + str(ssim) + '\n')
                else:
                    output_mem = output_mem.cpu().numpy()
                    result = np.argmax(output_mem, axis=1)
                    save_path = os.path.join(self.path, str(batch_idx) + '.png')
                    if self.dataset_name == 'DRIVE':
                        result = np.expand_dims(result, axis=3)
                        binary_img = recompone(result, 13, 12)
                        kill_border(binary_img, border_mask)
                        binary_img = np.squeeze(binary_img, axis=0)
                        if not eval_train:
                            cv2.imwrite(save_path, binary_img * 255)
                        pred_list.append(binary_img)
                    elif self.dataset_name == 'ISBI':
                        binary_img = np.squeeze(result, axis=0)
                        binary_img = np.expand_dims(binary_img, axis=2)
                        if not eval_train:
                            cv2.imwrite(save_path, binary_img * 255)
                        pred_list.append(binary_img)
                    elif self.dataset_name == 'CamSeq01':
                        result = np.squeeze(result, axis=0)
                        label = np.squeeze(label, axis=0)
                        rgb_img = onehot_to_rgb(result, self.id2code)
                        if not eval_train:
                            cv2.imwrite(save_path, rgb_img)
                        IOU.add_batch(result, label)
                        pred_list.append(rgb_img)

                if not eval_train:
                    acc_output = 0
                    for name, module in model.named_children():
                        if 'relu' in name:
                            acc_output += torch.sum(module.spikecount)
                    flops.append(acc_output.cpu().numpy())
        
        if self.dataset_name in ['CBSD', 'BSD', 'Set12']:
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list) 
        else:
            labels = np.stack(store_labels)
            if self.dataset_name in ['DRIVE', 'ISBI']:
                pred_imgs = np.stack(pred_list, axis=0)
                labels    = np.squeeze(labels)
                accuracy, jaccard_index, F1_score = seg.segmentation_index(pred_imgs, labels)
            else:
                _,_,_,miou,_ = IOU.evaluate()
        
        if eval_train:
            if self.dataset_name in ['CBSD', 'BSD', 'Set12']:
                return pred_list, avg_psnr, avg_ssim
            else:
                if self.dataset_name in ['DRIVE', 'ISBI']:
                    return pred_list, accuracy, jaccard_index, F1_score
                else:
                    return pred_list, miou
        else:
            if record_timestep:
                for str_time, flops in dict_flops_assumption.items():
                    flop = np.sum(flops) / ((batch_idx+1) * batch_size)
                    cost = flop * 0.9e-12 + flops_first * 4.6e-12
                    writer.add_scalar('flop', flop, int(str_time))
                    writer.add_scalar('flop_first', flops_first, int(str_time))
                    writer.add_scalar('cost', cost, int(str_time))

                if self.dataset_name in ['CBSD', 'BSD', 'Set12']:
                    for str_time, psnr in dict_store_psnr.items():
                        avg_psnr = sum(psnr) / len(psnr)
                        writer.add_scalar('psnr', avg_psnr, int(str_time))
                   
                    for str_time, ssim in dict_store_ssim.items():
                        avg_ssim = sum(ssim) / len(ssim)
                        writer.add_scalar('ssim', avg_ssim, int(str_time))
                else:
                    for str_time, results_list in dict_store_results.items():
                        results = np.stack(results_list, axis=0)
                        if self.dataset_name in ['DRIVE', 'ISBI']:
                            results = np.squeeze(results)
                            accuracy, jaccard_index, F1_score = seg.segmentation_index(results, labels)
                            writer.add_scalar('acc', accuracy,      int(str_time))
                            writer.add_scalar('JS',  jaccard_index, int(str_time))
                            writer.add_scalar('F1',  F1_score,      int(str_time))
                        else:
                            IOU = seg.IOUMetric(num_classes=32)
                            IOU.add_batch(results, labels)
                            _,_,_,miou,_ = IOU.evaluate()
                            writer.add_scalar('mIoU', miou, int(str_time))


            flop = np.sum(flops) / ((batch_idx+1) * batch_size)
            cost = flop * 0.9e-12 + flops_first * 4.6e-12
            if self.dataset_name in ['CBSD', 'BSD', 'Set12']:
                with open(os.path.join(self.path, 'results.txt'), 'w') as f:
                    f.write("PSNR of imgs is:" + str(avg_psnr) + '\n')
                    f.write("SSIM of imgs is:" + str(avg_ssim) + '\n')
                    f.write("FLOPS of imgs is:" + str(flop) + '\n')
                    f.write("Flop_first of imgs is:" + str(flops_first) + '\n')
                    f.write("Cost of imgs is:" + str(cost) + '\n')
                writer.add_text("eval", "PSNR of imgs is:" + str(avg_psnr) + '\n' + 
                                "SSIM of imgs is:" + str(avg_ssim) + '\n' +
                                "FlOPS of imgs is:" + str(flop) + '\n' + 
                                "Flop_first of imgs is:" + str(flops_first) + '\n'
                                "Cost of imgs is:" + str(cost) + '\n')

            else:
                if self.dataset_name in ['DRIVE', 'ISBI']:
                    with open(os.path.join(self.path, 'results.txt'), 'w') as f:
                        f.write('ACC is ' + str(accuracy) + '\n'
                                'JS is ' + str(jaccard_index) + '\n'
                                'F1 is ' + str(F1_score) + '\n')
                        f.write("FLOPS of imgs is:" + str(flop) + '\n')
                        f.write("flop_first of imgs is:" + str(flops_first) + '\n')
                        f.write("Cost of imgs is:" + str(cost) + '\n')
                    writer.add_text("eval", 'ACC is ' + str(accuracy) + '\n'
                                    'JS is ' + str(jaccard_index) + '\n'
                                    'F1 is ' + str(F1_score) + '\n'
                                    "FlOPS of imgs is:" + str(flop) + '\n' + 
                                    "Flop_first of imgs is:" + str(flops_first) + '\n' +
                                    "Cost of imgs is:" + str(cost) + '\n')
                else:
                    with open(os.path.join(self.path, 'results.txt'), 'w') as f:
                        f.write('miou is ' + str(miou) + '\n')
                        f.write("FLOPS of imgs is:" + str(flop) + '\n')
                        f.write("Flop_first of imgs is:" + str(flops_first) + '\n')
                        f.write("Cost of imgs is:" + str(cost) + '\n')
                    writer.add_text('eval', "mIoU is:" + str(miou) + '\n' +
                                    "FlOPS of imgs is:" + str(flop) + '\n' + 
                                    "Flop_first of imgs is:" + str(flops_first) + '\n' +
                                    "Cost of imgs is:" + str(cost) + '\n')

            writer.close()

    def set_state(self, modules, state):
        for _, module in modules._modules.items():
            if module._modules:
                self.set_state(module, state)
            else:
                if hasattr(module, 'set_state'):
                    module.set_state(state)

    def reset_snn(self, modules):
        for _, module in modules._modules.items():
            if module._modules:
                self.reset_snn(module)
            else:
                if hasattr(module, 'reset'):
                    module.reset()

