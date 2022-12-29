#type:ignore
import sys
from matplotlib import transforms
from thop import profile
from dataset_process.CamSeq_dataset import load_hdf5

sys.path.append("../")

import os
import cv2
import torch
import datetime
import numpy as np
from snn.tools.file_tools import check_exist_makedirs
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

input_activations = dict()
output_activations = dict()

def layer_name(name):
    def hook(module, input, output):
        input_activations[name] = input[0].clone().detach().cpu().numpy()
        output_activations[name] = output.clone().detach().cpu().numpy()
    return hook

def train_and_evaluate(net, device, data_path, output_path, train, dataset_name, input_channels, optimizer, epochs=40, batch_size=1, test=False):

    writer_path = os.path.join(output_path, 'logs', 'ann', dataset_name, str(input_channels))
    if dataset_name == 'Set12':
        model_save_path = os.path.join('./model/benli', 'BSD')
    else:
        model_save_path = os.path.join('./model/benli', dataset_name)
    evaluation_save_path = os.path.join(output_path, 'ann', dataset_name, str(input_channels))

    check_exist_makedirs(writer_path) 
    check_exist_makedirs(model_save_path)
    check_exist_makedirs(evaluation_save_path)

    writer = SummaryWriter(writer_path)

    if dataset_name == 'CamSeq01':
        from dataset_process.CamSeq_dataset import CamSeq_dataset, onehot_to_rgb
        from snn.tools.segmentation import IOUMetric
        train_dataset = CamSeq_dataset(data_path, train=True)
        test_dataset = CamSeq_dataset(data_path, train=False)
        id2code = test_dataset.id2code
        best_miou = 0.0
    elif dataset_name == 'ISBI':
        from dataset_process.ISBI_dataset import train_ISBI_dataset, test_ISBI_dataset
        from snn.tools.segmentation import segmentation_index
        train_datapath = os.path.join(data_path, 'train')
        test_datapath = os.path.join(data_path, 'test')
        train_dataset = train_ISBI_dataset(train_datapath)
        test_dataset = test_ISBI_dataset(test_datapath)
        best_F1_score = 0.0
    elif dataset_name == 'DRIVE':
        from dataset_process.DRIVE_dataset import DRIVE_dataset
        from snn.tools.segmentation import segmentation_index
        from snn.tools.segmentation.extract_patches import recompone, kill_border
        from snn.tools.segmentation.help_functions import load_hdf5
        train_dataset = DRIVE_dataset(data_path, train=True)
        test_dataset = DRIVE_dataset(data_path, train=False)
        test_border_masks = load_hdf5(os.path.join(data_path, 'DRIVE_dataset_borderMasks_test.hdf5'))
        best_F1_score = 0.0
    elif dataset_name == 'CBSD' or dataset_name == 'BSD':
        if dataset_name == 'CBSD':
            color = True
            prefix = "CBSD_patch_test_img_sigma_"
        else:
            color = False
            prefix = "BSD_patch_test_img_sigma_"
        from dataset_process.CBSD_dataset import train_CBSD_dataset, test_CBSD_dataset
        import snn.tools.denoising as denoise
        train_dataset = train_CBSD_dataset(data_path, color=color, valid=False, transform=transforms.ToTensor())
        test_dataset_15 = test_CBSD_dataset(data_path, prefix + "15.hdf5", color=color, transform=transforms.ToTensor())
        test_dataset_25 = test_CBSD_dataset(data_path, prefix + "25.hdf5", color=color, transform=transforms.ToTensor())
        test_dataset_35 = test_CBSD_dataset(data_path, prefix + "35.hdf5", color=color, transform=transforms.ToTensor())
        test_dataset_45 = test_CBSD_dataset(data_path, prefix + "45.hdf5", color=color, transform=transforms.ToTensor())
        test_dataset_50 = test_CBSD_dataset(data_path, prefix + "50.hdf5", color=color, transform=transforms.ToTensor())
        best_psnr = 0.0
    elif dataset_name == 'Set12':
        from dataset_process.Set_dataset import Set12_Dataset_hdf5
        test_dataset_15 = Set12_Dataset_hdf5(data_path='./data/benli/Set12_new/test', noise_level=15)
        test_dataset_25 = Set12_Dataset_hdf5(data_path='./data/benli/Set12_new/test', noise_level=25)
        test_dataset_35 = Set12_Dataset_hdf5(data_path='./data/benli/Set12_new/test', noise_level=35)
        test_dataset_45 = Set12_Dataset_hdf5(data_path='./data/benli/Set12_new/test', noise_level=45)
        test_dataset_50 = Set12_Dataset_hdf5(data_path='./data/benli/Set12_new/test', noise_level=50)
    else:
        pass

    if train:

        train_dataloader = DataLoader( 
            dataset=train_dataset,
            batch_size=batch_size, 
            shuffle=True
        )

    if dataset_name == 'DRIVE':
        test_dataloader = DataLoader(test_dataset, batch_size=156)
    elif dataset_name in ['CBSD', 'BSD']:
        test_dataloader_15 = DataLoader(
            test_dataset_15,
            batch_size=4
        )
        test_dataloader_25 = DataLoader(
            test_dataset_25,
            batch_size=4
        )
        test_dataloader_35 = DataLoader(
            test_dataset_35,
            batch_size=4
        )
        test_dataloader_45 = DataLoader(
            test_dataset_45,
            batch_size=4
        )
        test_dataloader_50 = DataLoader(
            test_dataset_50,
            batch_size=4
        )
    elif dataset_name == 'Set12':
        test_dataloader_15 = DataLoader(
            test_dataset_15,
            batch_size=1
        )
        test_dataloader_25 = DataLoader(
            test_dataset_25,
            batch_size=1
        )
        test_dataloader_35 = DataLoader(
            test_dataset_35,
            batch_size=1
        )
        test_dataloader_45 = DataLoader(
            test_dataset_45,
            batch_size=1
        )
        test_dataloader_50 = DataLoader(
            test_dataset_50,
            batch_size=1
        )

    else:
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size
        )

    if train:
        # 定义Loss算法  
        if dataset_name not in ["Set", "BSD", "CBSD"]:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.L1Loss()
        start = datetime.datetime.now() 
        # 训练epochs次  
        print("training time")
        for epoch in tqdm(range(epochs)):
            if test:
                if epoch == 1:
                    break
            # 训练模式
            net.train()
            running_loss = 0.0
            # 按照batch_size开始训练
            for image, label in train_dataloader:
                optimizer.zero_grad()
                # 将数据拷贝到device中
                if dataset_name not in ["Set", "BSD", "CBSD"]:
                    image = image.to(device=device)
                    label = label.to(device=device, dtype = torch.long)
                    label = torch.squeeze(label, dim=1)
                else:
                    image = image.to(device=device, dtype = torch.float32)
                    label = label.to(device, dtype = torch.float32)

                # 使用网络参数，输出预测结果
                pred = net(image)
                # 计算loss
                loss = criterion(pred, label)
                running_loss += loss
                # 更新参数
                loss.backward()
                optimizer.step()

                if dataset_name == 'CamSeq01':
                    iou_evaluator = IOUMetric(num_classes=32)
                    with torch.no_grad():
                        for idx, (data, label) in enumerate(test_dataloader):
                            data = data.to(device, dtype = torch.float32)
                            batch_pred_img = net.forward(data, 'original').cpu().numpy()
                            batch_pred_img = np.argmax(batch_pred_img, axis = 1)
                            gtruth_mask = label.numpy()
                            iou_evaluator.add_batch(batch_pred_img, gtruth_mask)
                        acc, acc_cls, iu, miou, fwavacc = iou_evaluator.evaluate()
                        if miou > best_miou:
                            best_miou = miou
                            torch.save(net.state_dict(), os.path.join(model_save_path, dataset_name + '.pth'))
                elif dataset_name == 'ISBI':
                    pred_imgs = []
                    gtruth_masks = []
                    with torch.no_grad():
                        for idx, (data, label) in enumerate(test_dataloader):
                            data = data.to(device, dtype = torch.float32)
                            batch_pred_img = net.forward(data, 'original').cpu().numpy()
                            batch_pred_img = np.argmax(batch_pred_img, axis = 1)
                            batch_pred_img = np.squeeze(batch_pred_img)
                            gtruth_mask = np.squeeze(label.numpy())
                            pred_imgs.append(batch_pred_img)
                            gtruth_masks.append(gtruth_mask)
                    pred_imgs = np.stack(pred_imgs, axis=0)
                    gtruth_masks = np.stack(gtruth_masks, axis=0)
                    accuracy, jaccard_index, F1_score = segmentation_index(pred_imgs, gtruth_masks)
                    if F1_score > best_F1_score:
                        best_F1_score = F1_score
                        torch.save(net.state_dict(), os.path.join(model_save_path, dataset_name + '.pth'))

                elif dataset_name == 'DRIVE_C':
                    pred_imgs = []
                    gtruth_masks = []
                    with torch.no_grad():
                        for idx, (data, label) in enumerate(test_dataloader):
                            data = data.to(device, dtype = torch.float32)
                            batch_pred_img = net.forward(data, 'original').cpu().numpy()
                            batch_pred_img = np.argmax(batch_pred_img, axis = 1)
                            batch_pred_img = np.squeeze(batch_pred_img)
                            gtruth_mask = np.squeeze(label.numpy())
                            pred_imgs.append(batch_pred_img)
                            gtruth_masks.append(gtruth_mask)
                    pred_imgs = np.stack(pred_imgs, axis=0)
                    gtruth_masks = np.stack(gtruth_masks, axis=0)
                    accuracy, jaccard_index, F1_score = segmentation_index(pred_imgs, gtruth_masks)
                    if F1_score > best_F1_score:
                        best_F1_score = F1_score
                        torch.save(net.state_dict(), os.path.join(model_save_path, dataset_name + '.pth'))

            if dataset_name in ['BSD', 'CBSD']:
                img_psnr = []
                with torch.no_grad():
                    for idx, (data, label) in enumerate(test_dataloader_25):
                        data = data.to(device, dtype=torch.float32)
                        batch_pred_img = net.forward(data, 'original')
                        batch_pred_img = batch_pred_img.permute(0,2,3,1).cpu().numpy()
                        label = label.permute(0,2,3,1).cpu().numpy()
                        label = label * 255.
                            
                        data = data.cpu().numpy()
                        _, psnr, ssim = denoise.merge_and_psnr(orignal=data, noise=batch_pred_img, groundtruth=label, patches=4, poisson = False, save_flag=0)
                        img_psnr.append(psnr)
                avg_psnr = sum(img_psnr) / len(img_psnr)
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save(net.state_dict(), os.path.join(model_save_path, version + '.pth'))

            if dataset_name == 'DRIVE':
                pred_imgs = []
                gtruth_masks = []
                with torch.no_grad():
                    for idx, (data, label) in enumerate(test_dataloader):
                        data = data.to(device, dtype = torch.float32)
                        test_border_mask = test_border_masks[range(idx, idx + 1)]
                        pred_patches = net.forward(data, 'original').cpu().numpy()
                        pred_patches = np.argmax(pred_patches, axis=1)
                        pred_patches = np.expand_dims(pred_patches, axis=3)
                        label_patches = np.expand_dims(label, axis=3)
                        pred_img = recompone(pred_patches, 13, 12)
                        gtruth_mask = recompone(label_patches, 13, 12)
                        kill_border(pred_img, test_border_mask)
                        kill_border(gtruth_mask, test_border_mask)
                        pred_imgs.append(pred_img)
                        gtruth_masks.append(gtruth_mask)
                pred_imgs = np.stack(pred_imgs, axis=0)
                gtruth_masks = np.stack(gtruth_masks, axis=0)
                accuracy, jaccard_index, F1_score = segmentation_index(pred_imgs, gtruth_masks)
                if F1_score > best_F1_score:
                    best_F1_score = F1_score
                    torch.save(net.state_dict(), os.path.join(model_save_path, version + '.pth'))

            if dataset_name not in ["Set", "BSD", "CBSD"]:
                if dataset_name == 'CamSeq01':
                    writer.add_scalar("evaluate miou", best_miou, epoch)
                else:
                    writer.add_scalar('evaluate F1_score', best_F1_score, epoch)
                writer.add_scalar("training loss", running_loss / len(train_dataloader) , epoch) 
            else:
                writer.add_scalar("evaluation PSNR", best_psnr, epoch)
                writer.add_scalar("training loss", running_loss / len(train_dataloader) , epoch) 
    
        end = datetime.datetime.now()
        time_mean = ((end - start) / epochs).seconds
        format_time = datetime.timedelta(seconds=time_mean)
        print("each epoch mean time: {}".format(format_time))
        writer.add_text("training", "each epoch mean time:{}".format(format_time))
        print("training done")

    print('evaluation time')
    if dataset_name == 'Set12':
        net.load_state_dict(torch.load(os.path.join(model_save_path, 'BSD.pth')))
    else:
        net.load_state_dict(torch.load(os.path.join(model_save_path, dataset_name + '.pth'), map_location='cpu'))

    if dataset_name == 'ISBI':
        start = datetime.datetime.now()
        pred_imgs = []
        gtruth_masks = []
        with torch.no_grad():
            for idx, (data, label) in enumerate(test_dataloader):
                data = data.to(device, dtype = torch.float32)
                if idx == 0:
                    one_data = data[range(0,1),:,:,:]
                    flops, params = profile(net, (one_data,'original',))
                    cost = flops * 4.6e-12

                batch_pred_img = net.forward(data, 'original').cpu().numpy()
                batch_pred_img = np.argmax(batch_pred_img, axis = 1)
                for index in range(batch_pred_img.shape[0]):
                    binary_img = batch_pred_img[index] * 255
                    binary_img = np.expand_dims(binary_img, axis=2)
                    label_img = label[index] * 255
                    result_path = os.path.join(evaluation_save_path, str(idx * batch_size + index) + '.png')
                    cv2.imwrite(result_path, binary_img)
                    writer.add_image(str(idx * batch_size + index) + "_data", data[index])
                    writer.add_image(str(idx * batch_size + index) + "_originals", label_img / 255)
                    writer.add_image(str(idx * batch_size + index) + "_results", binary_img / 255, dataformats='HWC')

                batch_pred_img = np.squeeze(batch_pred_img)
                gtruth_mask = np.squeeze(label.numpy())
                pred_imgs.append(batch_pred_img)
                gtruth_masks.append(gtruth_mask)
            
            end = datetime.datetime.now()
            time_diff = ((end - start)).seconds
            format_time = datetime.timedelta(seconds=time_diff)
            print("evaluation time: {}".format(format_time))
            writer.add_text("testing", "evaluation time:{}".format(format_time))
            pred_imgs = np.stack(pred_imgs, axis=0)
            gtruth_masks = np.stack(gtruth_masks, axis=0)
            accuracy, jaccard_index, F1_score = segmentation_index(pred_imgs, gtruth_masks)
            print("acc is:{}, JS is:{}, F1 is {}".format(accuracy, jaccard_index, F1_score))
            writer.add_text("testing", "acc is:{}, JS is:{}, F1 is {}, flop is {}, cost is {}J".format(accuracy, jaccard_index, F1_score, flops, cost))
            with open(os.path.join(evaluation_save_path, 'result.txt'), 'w') as f:
                f.write('acc is ' + str(accuracy) + '\n'
                        'JS is ' + str(jaccard_index) + '\n'
                        'F1 is ' + str(F1_score) + '\n'
                        'flop is ' + str(flops) + '\n'
                        'cost is ' + str(cost) + 'J' + '\n') 

        # for i in range(1000):
        #     cost = flops * 4.6e-12
        #     writer.add_scalar('cost_pJ',cost, i)

    elif dataset_name == 'DRIVE':
        start = datetime.datetime.now()
        pred_imgs = []
        gtruth_masks = []
        with torch.no_grad():
            for idx, (data, label) in enumerate(test_dataloader):
                data = data.to(device, dtype = torch.float32)
                if idx == 0:
                    one_data = data[range(0,1),:,:,:]
                    flops, params = profile(net, (one_data,'original',))
                    cost = flops * 4.6e-12
                # flops, params = profile(net, (data,))
                pred_patches = net.forward(data, 'original').cpu().numpy()
                test_border_mask = test_border_masks[range(idx, idx + 1)]
                pred_patches = np.argmax(pred_patches, axis = 1)
                pred_patches = np.expand_dims(pred_patches, axis=3)
                label_patches = np.expand_dims(label, axis=3)
                pred_img = recompone(pred_patches, 13, 12)
                gtruth_mask = recompone(label_patches, 13, 12)
                kill_border(pred_img, test_border_mask)
                kill_border(gtruth_mask, test_border_mask)
                result_path = os.path.join(evaluation_save_path, str(idx) + '.png')
                gt = np.squeeze(gtruth_mask, axis=0)
                result = np.squeeze(pred_img, axis=0)
                cv2.imwrite(result_path, result * 255)
                writer.add_image(str(idx) + "_originals", gt, dataformats='HWC')
                writer.add_image(str(idx) + "_results", result, dataformats='HWC')
                pred_imgs.append(pred_img)
                gtruth_masks.append(gtruth_mask)
            
            end = datetime.datetime.now()
            time_diff = ((end - start)).seconds
            format_time = datetime.timedelta(seconds=time_diff)
            print("evaluation time: {}".format(format_time))
            writer.add_text("testing", "evaluation time:{}".format(format_time))
            pred_imgs = np.stack(pred_imgs, axis=0)
            gtruth_masks = np.stack(gtruth_masks, axis=0)
            accuracy, jaccard_index, F1_score = segmentation_index(pred_imgs, gtruth_masks)
            print("acc is:{}, JS is:{}, F1 is {}".format(accuracy, jaccard_index, F1_score))
            writer.add_text("testing", "acc is:{}, JS is:{}, F1 is {}, flop is {}, cost is {}J".format(accuracy, jaccard_index, F1_score, flops, cost))
            with open(os.path.join(evaluation_save_path, 'result.txt'), 'w') as f:
                f.write('acc is ' + str(accuracy) + '\n'
                        'JS is ' + str(jaccard_index) + '\n'
                        'F1 is ' + str(F1_score) + '\n'
                        'flop is ' + str(flops) + '\n'
                        'cost is ' + str(cost) + 'J' + '\n') 

        # for i in range(1000):
        #     cost = flops * 4.6e-12
        #     writer.add_scalar('cost_pJ',cost, i)
    
    elif dataset_name == 'CamSeq01':
        iou_evaluator = IOUMetric(num_classes=32)
        # layer_list = ['relu11', 'relu13', 'relu15', 'relu17']
        # layer_list = ['relu17']
        # layer_list = ['conv17']
        # for name, module in net.named_modules():
        #     if name in layer_list:
        #         module.register_forward_hook(layer_name(name))

        start = datetime.datetime.now()
        with torch.no_grad():
            for idx, (data, label) in enumerate(test_dataloader):
                data = data.to(device, dtype = torch.float32)
                if idx == 0:
                    one_data = data[range(0,1),:,:,:]
                    flops, params = profile(net, (one_data,'original',))
                    cost = flops * 4.6e-12
                # flops, params = profile(net, (data,))
                batch_pred_img = net.forward(data, 'original').cpu().numpy()
                # np.save('./activation/ann/CamSeq01/ann_input.npy', input_activations)
                # exit(0)
                
                batch_pred_img = np.argmax(batch_pred_img, axis = 1)
                gtruth_mask = label.numpy()
                iou_evaluator.add_batch(batch_pred_img, gtruth_mask)
                for index in range(batch_pred_img.shape[0]):
                    rgb_img = onehot_to_rgb(batch_pred_img[index], id2code)
                    label_img = onehot_to_rgb(label[index], id2code)
                    result_path = os.path.join(evaluation_save_path, str(idx * batch_size + index) + '.png')
                    cv2.imwrite(result_path, rgb_img)
                    writer.add_image(str(idx * batch_size + index) + "_data", data[index])
                    writer.add_image(str(idx * batch_size + index) + "_originals", label_img, dataformats='HWC')
                    writer.add_image(str(idx * batch_size + index) + "_results", rgb_img, dataformats='HWC')
            
            end = datetime.datetime.now()
            time_diff = ((end - start)).seconds
            format_time = datetime.timedelta(seconds=time_diff)
            print("evaluation time: {}".format(format_time))
            writer.add_text("testing", "evaluation time:{}".format(format_time))
            
            acc, acc_cls, iu, miou, fwavacc = iou_evaluator.evaluate()
            print("evaluation miou:{}".format(miou))
            writer.add_text("testing", "evaluation miou:{}, flop is {}, cost is {}J".format(miou, flops, cost))
            with open(os.path.join(evaluation_save_path, 'result.txt'), 'w') as f:
                f.write("evaluation miou:{}, flop is {}, cost is {}J".format(miou, flops, cost))
        # for i in range(1000):
        #     cost = flops * 4.6e-12
        #     writer.add_scalar('cost_pJ',cost, i)
    
    elif dataset_name in ['BSD', 'CBSD', 'Set12']:
        start = datetime.datetime.now()
        eval_15_path = os.path.join(evaluation_save_path, "sigma_15")
        avg_psnr_15, avg_ssim_15, flop_15, cost_15 = denoise_eval(eval_15_path, net, dataset_name, test_dataloader_15, device, test)

        end = datetime.datetime.now()
        time_diff = ((end - start)).seconds
        format_time = datetime.timedelta(seconds=time_diff)
        print("evaluation time: {}".format(format_time))

        eval_25_path = os.path.join(evaluation_save_path, "sigma_25")
        avg_psnr_25, avg_ssim_25, flop_25, cost_25  = denoise_eval(eval_25_path, net, dataset_name, test_dataloader_25, device, test)

        eval_35_path = os.path.join(evaluation_save_path, "sigma_35")
        avg_psnr_35, avg_ssim_35, flop_35, cost_35  = denoise_eval(eval_35_path, net, dataset_name, test_dataloader_35, device, test)

        eval_45_path = os.path.join(evaluation_save_path, "sigma_45")
        avg_psnr_45, avg_ssim_45, flop_45, cost_45  = denoise_eval(eval_45_path, net, dataset_name, test_dataloader_45, device, test)

        eval_50_path = os.path.join(evaluation_save_path, "sigma_50")
        avg_psnr_50, avg_ssim_50, flop_50, cost_50 = denoise_eval(eval_50_path, net, dataset_name, test_dataloader_50, device, test)
        
        writer.add_text("testing", "evaluation time:{}".format(format_time))
        writer.add_text("testing", "evaluation psnr 15:{}, ssim 15:{}, psnr 25:{}, ssim 25:{}, psnr 50:{}, ssim 50:{}".format(avg_psnr_15, avg_ssim_15, 
                                                                                                                              avg_psnr_25, avg_ssim_25,
                                                                                                                              avg_psnr_50, avg_ssim_50))
        with open(os.path.join(eval_15_path, 'result.txt'), 'w') as f:
            f.write("evaluation psnr:{}, ssim:{}, flop:{}, cost:{}J".format(avg_psnr_15, avg_ssim_15, flop_15, cost_15))
        with open(os.path.join(eval_25_path, 'result.txt'), 'w') as f:
            f.write("evaluation psnr:{}, ssim:{}, flop:{}, cost:{}J".format(avg_psnr_25, avg_ssim_25, flop_25, cost_25))
        with open(os.path.join(eval_35_path, 'result.txt'), 'w') as f:
            f.write("evaluation psnr:{}, ssim:{}, flop:{}, cost:{}J".format(avg_psnr_35, avg_ssim_35, flop_35, cost_35))
        with open(os.path.join(eval_45_path, 'result.txt'), 'w') as f:
            f.write("evaluation psnr:{}, ssim:{}, flop:{}, cost:{}J".format(avg_psnr_45, avg_ssim_45, flop_45, cost_45))
        with open(os.path.join(eval_50_path, 'result.txt'), 'w') as f:
            f.write("evaluation psnr:{}, ssim:{}, flop:{}, cost:{}J".format(avg_psnr_50, avg_ssim_50, flop_50, cost_50))

    writer.close()

def denoise_eval(path, net, dataset_name, test_dataloader, device, test):
    import snn.tools.denoising as denoise 
    import matplotlib.pyplot as plt
    img_psnr = []
    img_ssim = []
    layer_list = ['conv17']
    for name, module in net.named_modules():
        if name in layer_list:
            module.register_forward_hook(layer_name(name))

    with torch.no_grad():
        kwargs = {"pic_path": path}
        check_exist_makedirs(path)
        for idx, (data, label) in enumerate(test_dataloader):
            if test:
                if idx == 1:
                    break
            data = data.to(device, dtype=torch.float32)
            if idx == 0:
                one_data = data[range(0,1),:,:,:]
                flops, params = profile(net, (one_data,'original',))
                cost = flops * 4.6e-12

            batch_pred_img = net.forward(data, 'original')
            # np.save('./activation/ann/BSD/ann_input.npy', input_activations)
            # exit(0)

            batch_pred_img = batch_pred_img.permute(0,2,3,1).cpu().numpy()
            label = label.permute(0,2,3,1).cpu().numpy()
            if dataset_name != 'Set12':
                label = label * 255.
            # plt_data = data.permute(0,2,3,1).cpu().numpy()

            # plt.imshow(plt_data[0])
            # plt.savefig('./ann_ori_0.png')
            # plt.imshow(batch_pred_img[0])
            # plt.colorbar()
            # plt.savefig('./ann_fig_0.png')
            if dataset_name == 'Set12':
                _, psnr, ssim = denoise.image_save_psnr(pic_path=path,
                                                        orignal=data, 
                                                        noise=batch_pred_img,
                                                        groundtruth=label,
                                                        batch_idx=idx,
                                                        save_flag=True)
            else:
                _, psnr, ssim = denoise.merge_and_psnr(pic_path=path,
                                                       orignal=data,
                                                       noise=batch_pred_img,
                                                       groundtruth=label, 
                                                       patches=4, 
                                                       batch_idx=idx, 
                                                       poisson=False, 
                                                       save_flag=True) 
            img_psnr.append(psnr)
            img_ssim.append(ssim)

    avg_psnr = sum(img_psnr) / len(img_psnr)
    avg_ssim = sum(img_ssim) / len(img_ssim)
    
    return avg_psnr, avg_ssim, flops, cost
