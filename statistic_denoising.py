from os import path as osp
import numpy as np
import json
from os import path as osp
from scipy import stats

def return_PSNR_SSIM(path):
  with open(osp.join(path, 'result_every_img.txt')) as f:
    PSNR, SSIM = [], []
    for line in f.readlines():
      split_list = line.split(' ')
      PSNR.append(float(split_list[1].split(':')[1]))
      SSIM.append(float(split_list[2].split(':')[1].strip()))
  return PSNR, SSIM

def get_p_value(MTF_performance, other_performance):
  # 检查数据正态性
  _, p_normal_model1 = stats.shapiro(MTF_performance)
  _, p_normal_model2 = stats.shapiro(other_performance)

  print("Model 1 normality test p-value:", p_normal_model1)
  print("Model 2 normality test p-value:", p_normal_model2)

  # 如果两个样本都符合正态分布，可以使用t检验
  if p_normal_model1 > 0.05 and p_normal_model2 > 0.05:
      _, p_value = stats.ttest_ind(MTF_performance, other_performance, alternative='greater')
      print("Independent t-test p-value:", p_value)
  else:
      _, p_value = stats.mannwhitneyu(MTF_performance, other_performance, alternative='greater')
      print("mannwhitneyu检验来比较配对数据 test p-value:", p_value)
  return p_value

neurons = ['MLF', 'MTF', 'Real_IF']
neuron_character = {}
# dataset = 'BSD68'
dataset = 'CBSD68'
path = osp.join('.', dataset)
for neuron in neurons:
    target_path = osp.join(path, neuron)
    psnr, ssim = return_PSNR_SSIM(target_path)
    neuron_character[neuron] = {'PSNR':psnr, 'SSIM':ssim}
test_neurons = ['MLF', 'Real_IF']
store_p_value = {'PSNR':{}, 'SSIM':{}}

for test_neuron in test_neurons:
    PSNR_p_value = get_p_value(neuron_character['MTF']['PSNR'], neuron_character[test_neuron]['PSNR'])
    SSIM_p_value = get_p_value(neuron_character['MTF']['SSIM'], neuron_character[test_neuron]['SSIM'])
    store_p_value['PSNR'][test_neuron] = PSNR_p_value
    store_p_value['SSIM'][test_neuron] = SSIM_p_value

with open(osp.join('.', dataset + '.json'), 'w') as f:
    json.dump(store_p_value, f)