import cv2
from os import path as osp
import glob
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from scipy import stats
import json

def segmentation_index(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    confusion = confusion_matrix(y_true, y_pred)
    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))

    jaccard_index = jaccard_score(y_true, y_pred)
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    return accuracy, jaccard_index, F1_score

def return_all_metric(path, label_base_path):
    acc, JS, F1 = [], [], []
    for sub_dir in sorted(os.listdir(path)):
        sub_path = osp.join(path, sub_dir)
        png_list = sorted(glob.glob(osp.join(sub_path, "*.png")))
        for png_path in png_list:
            label_path = osp.join(label_base_path, osp.basename(png_path))
            image = cv2.imread(png_path, -1)
            label = cv2.imread(label_path, -1)
            image[image==255] = 1
            label[label==255] = 1
            accuracy, jaccard_index, F1_score = segmentation_index(image, label)
            acc.append(accuracy)
            JS.append(jaccard_index)
            F1.append(F1_score)
    return acc, JS, F1

def get_p_value(MTF_performance, other_performance):
  # 检查数据正态性
  _, p_normal_model1 = stats.shapiro(MTF_performance)
  _, p_normal_model2 = stats.shapiro(other_performance)

  # 如果两个样本都符合正态分布，可以使用t检验
  if p_normal_model1 > 0.05 and p_normal_model2 > 0.05:
      _, p_value = stats.ttest_ind(MTF_performance, other_performance, alternative='greater')
      print("Independent t-test p-value:", p_value)
  else:
      _, p_value = stats.mannwhitneyu(MTF_performance, other_performance, alternative='greater')

      print("mannwhitneyu检验来比较配对数据 test p-value:", p_value)
  return p_value


dataset = './DRIVE'
prefix_path = dataset
label_base_path = osp.join(dataset, 'label')

neuron = ['MLF', 'Real_IF', 'MTF']
result_dict = {}
for test_neuron in neuron:
    result_dict[test_neuron] = {'acc':[], 'JS':[], 'F1':[]}
    acc, JS, F1 = 0, 0, 0
    path = osp.join(prefix_path, test_neuron)
    png_list = glob.glob(osp.join(path, '*.png'))
    for png_path in png_list:
        label_path = osp.join(label_base_path, osp.basename(png_path))
        image = cv2.imread(png_path, -1)
        label = cv2.imread(label_path, -1)
        image[image==255] = 1
        label[label==255] = 1
        accuracy, jaccard_index, F1_score = segmentation_index(image, label)
        result_dict[test_neuron]['acc'].append(accuracy)
        result_dict[test_neuron]['JS'].append(jaccard_index)
        result_dict[test_neuron]['F1'].append(F1_score)

neuron_character = result_dict
path = osp.join('.', dataset)
test_neurons = ['MLF', 'Real_IF']
store_p_value = {'acc':{}, 'JS':{}, 'F1':{}}

for test_neuron in test_neurons:
    acc_p_value = get_p_value(neuron_character['MTF']['acc'], neuron_character[test_neuron]['acc'])
    JS_p_value = get_p_value(neuron_character['MTF']['JS'], neuron_character[test_neuron]['JS'])
    F1_p_value = get_p_value(neuron_character['MTF']['F1'], neuron_character[test_neuron]['F1'])

    store_p_value['acc'][test_neuron] = acc_p_value
    store_p_value['JS'][test_neuron] = JS_p_value
    store_p_value['F1'][test_neuron] = F1_p_value

with open(osp.join('.', dataset + '.json'), 'w') as f:
    json.dump(store_p_value, f)


## CamSeq01
# num_classes = 32
# def compute_IoU(label_pred, label_true):
#     hist = np.zeros((num_classes, num_classes))
#     label_pred, label_true = label_pred.flatten(), label_true.flatten()
#     mask = (label_true >= 0) & (label_true < num_classes)
#     hist = np.bincount(
#         num_classes * label_true[mask].astype(int) +
#         label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
#     acc_cls = np.diag(hist) / hist.sum(axis=1)
#     acc_cls = np.nanmean(acc_cls)
#     IoU = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

#     return IoU

# def get_p_value(MTF_performance, other_performance):
#   # 检查数据正态性
#   _, p_normal_model1 = stats.shapiro(MTF_performance)
#   _, p_normal_model2 = stats.shapiro(other_performance)

#   print("Model 1 normality test p-value:", p_normal_model1)
#   print("Model 2 normality test p-value:", p_normal_model2)

#   # 如果两个样本都符合正态分布，可以使用t检验
#   if p_normal_model1 > 0.05 and p_normal_model2 > 0.05:
#       _, p_value = stats.ttest_ind(MTF_performance, other_performance, alternative='greater')
#       print("Independent t-test p-value:", p_value)
#   else:
#       _, p_value = stats.mannwhitneyu(model1_performance, model2_performance)
#       print("mannwhitneyu检验来比较配对数据 test p-value:", p_value)
#   return p_value

# def parse_code(l):
#     '''Function to parse lines in a text file, returns separated elements (label codes and names in this case)
#     '''
#     if len(l.strip().split("\t")) == 2:
#         a, b = l.strip().split("\t")
#         return tuple(int(i) for i in a.split(' ')), b
#     else:
#         a, b, c = l.strip().split("\t")
#         return tuple(int(i) for i in a.split(' ')), c

# def rgb_to_onehot(rgb_image, colormap):
#     '''Function to one hot encode RGB mask labels
#         Inputs:
#             rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
#             colormap - dictionary of color to label id
#         Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
#     '''
#     num_classes = len(colormap)
#     shape = rgb_image.shape[:2]+(num_classes,)
#     encoded_image = np.zeros(shape, dtype=np.int8 )
#     for i, cls in enumerate(colormap):
#         encoded_image[:,:,i] = np.all(rgb_image.reshape((-1,3)) == colormap[i], axis=1).reshape(shape[:2])
#     return encoded_image

# def calculate_iou_per_class(true_labels, pred_labels, num_classes=32):
#     iou_scores = []

#     for class_id in range(num_classes):
#         # 确定每个类别的真实和预测区域
#         true_class = (true_labels == class_id)
#         pred_class = (pred_labels == class_id)

#         # 计算交集和并集
#         intersection = np.logical_and(true_class, pred_class)
#         union = np.logical_or(true_class, pred_class)

#         # 避免除以零
#         union_sum = np.sum(union)
#         if union_sum == 0:
#             iou = float('nan')  # 或者根据需要设置为0或1
#         else:
#             iou = np.sum(intersection) / union_sum

#         iou_scores.append(iou)

#     return iou_scores

# def return_iou(path, label_base_path, id2code):
#     IoU_list = []
#     png_list = sorted(glob.glob(osp.join(path, "*.png")))
#     for png_path in png_list:
#         label_path = osp.join(label_base_path, osp.basename(png_path))
#         image = cv2.imread(png_path, -1)
#         label = cv2.imread(label_path, -1)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
#         image = np.argmax(rgb_to_onehot(image, id2code),axis=2)
#         label = np.argmax(rgb_to_onehot(label, id2code),axis=2)

#         IoU = calculate_iou_per_class(image, label)
#         IoU_list.append(np.nanmean(IoU))

#     return IoU_list

# neurons = ['MLF', 'MTF', 'Real_IF']
# neuron_character = {}
# dataset = 'CamSeq01'
# path = osp.join('.', dataset)

# label_codes, label_names = zip(*[parse_code(l) for l in open(os.path.join(path, "label_colors.txt"))])
# label_codes, label_names = list(label_codes), list(label_names)
# code2id = {v:k for k,v in enumerate(label_codes)}
# id2code = {k:v for k,v in enumerate(label_codes)}
# name2id = {v:k for k,v in enumerate(label_names)}
# id2name = {k:v for k,v in enumerate(label_names)}

# for neuron in neurons:
#     target_path = osp.join(path, neuron)
#     IoU_list = return_iou(target_path, osp.join(path, 'label'), id2code)
#     neuron_character[neuron] = {'IoU': IoU_list}
# test_neurons = ['MLF', 'Real_IF']
# store_p_value = {'IoU':{}}

# for test_neuron in test_neurons:
#     IoU_p_value = get_p_value(neuron_character['MTF']['IoU'], neuron_character[test_neuron]['IoU'])
#     store_p_value['IoU'][test_neuron] = IoU_p_value

# with open(osp.join('.', dataset + '.json'), 'w') as f:
#     json.dump(store_p_value, f)
