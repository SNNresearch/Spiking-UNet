import cv2
import numpy as np
import os 

def psnr(img1, img2, border=0):
    import math
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def merge_and_psnr(pic_path=None, orignal=None, noise=None, groundtruth=None, patches=4, batch_idx=0, poisson=False, save_flag=False): 
    if save_flag:
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)  
        img_result_file = os.path.join(pic_path, "result_every_img.txt")
        
        
    orignal = orignal.permute(0,2,3,1)
    orignal = orignal.cpu().numpy()
    sf = 1
    h, w = groundtruth.shape[1:-1]
    pred_patches = noise.shape[0]
    if poisson:
        pred = noise
    else:
        pred = []
        for index in range(pred_patches):
            noise_img = noise[index]
            pred_img = orignal[index] - noise_img
            pred.append(pred_img)
    pred = np.array(pred)
    img_num = pred_patches // patches
    prediction = np.zeros((img_num, sf * h, sf * w, groundtruth.shape[-1]))
    for img_index in range(img_num):
        prediction[img_index,:h//2*sf, :w//2*sf, :] = pred[img_index * patches + 0][:h//2*sf, :w//2*sf, :]
        prediction[img_index,:h//2*sf, w//2*sf:w*sf, :] = pred[img_index * patches + 1][:h//2*sf, (-w + w//2)*sf:, :]
        prediction[img_index,h//2*sf:h*sf, :w//2*sf, :] = pred[img_index * patches + 2][(-h + h//2)*sf:, :w//2*sf, :]
        prediction[img_index,h//2*sf:h*sf, w//2*sf:w*sf, :] = pred[img_index * patches + 3][(-h + h//2)*sf:, (-w + w//2)*sf:, :]
        prediction[img_index] = prediction[img_index].clip(0,1) * 255
        if save_flag:
            cv2.imwrite(os.path.join(pic_path, 'prediction' + str(batch_idx + img_index) + '.png'), prediction[img_index][...,::-1])
            # cv2.imwrite(os.path.join(pic_path, 'groundtruth' + str(batch_idx + img_index) + '.png'), groundtruth[img_index][...,::-1] * 255)
        img_psnr = psnr(img1 = prediction[img_index], img2= groundtruth[img_index])
        img_ssim = calculate_ssim(img1 = prediction[img_index], img2= groundtruth[img_index])
    prediction = prediction[...,::-1]

    return prediction, img_psnr, img_ssim

def image_save_psnr(pic_path, orignal, noise, groundtruth, batch_idx, save_flag=False):
    if save_flag:
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)  
    # img_result_file = os.path.join(pic_path, "result_every_img.txt")
    orignal = orignal.permute(0,2,3,1)
    orignal = orignal.cpu().numpy()
    prediction = (orignal - noise)[0]
    prediction = prediction.clip(0,1) * 255
    groundtruth = groundtruth[0]
    if save_flag:
        cv2.imwrite(os.path.join(pic_path, 'prediction' + str(batch_idx) + '.png'), prediction)
    img_psnr = psnr(img1 = prediction, img2= groundtruth)
    img_ssim = calculate_ssim(img1 = prediction, img2= groundtruth)
    # with open(img_result_file, 'a+') as f:
    #     f.write(str(batch_idx) + "psnr is :" + str(img_psnr) + '\n')
    #     f.write(str(batch_idx) + "ssim is :" + str(img_ssim) + '\n')
    return prediction, img_psnr, img_ssim
