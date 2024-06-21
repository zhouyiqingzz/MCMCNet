import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.options import Opts
from models.Our_Model import DARNet
import torch
import os
from PIL import Image
import random
from datasets import custom_transform as tr
from torchvision.transforms import transforms
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('../') #目的是python解释器找到上级目录，若是上两级则换为sys.path.append('../../')
random.seed(3407)

opts = Opts()
model = DARNet(opts)#.cuda()
model=model.eval()
model_path='/home/arsc/tmp/pycharm_project_698/DA_Road/run_model_experiments/our_model/CHN6/experiment_20240611_111621/120_checkpoint.pth'
# model_path='/home/arsc/tmp/pycharm_project_698/DA_Road/run_model_experiments/our_model/CHN6/experiment_20240616_070449/12_checkpoint.pth'
checkpoint = torch.load(model_path, map_location='cpu')
# print(checkpoint['state_dict'])
for key,param in list(checkpoint['state_dict'].items()):
    if key.startswith('module.'):
        checkpoint['state_dict'][key[7:]]=param
        checkpoint['state_dict'].pop(key)
model.load_state_dict(checkpoint['state_dict'], strict=True)

img_path = os.path.join('/home/arsc/tmp/pycharm_project_639/CSA/data/CHN6/val/images/am100785_sat.jpg')
label_path = os.path.join('/home/arsc/tmp/pycharm_project_639/CSA/data/CHN6/val/gt/am100785_mask.png')
ske_path = os.path.join('/home/arsc/tmp/pycharm_project_639/CSA/data/CHN6/val/skeleton/am100785_mask.png')
image, gt, ske = Image.open(img_path).convert('RGB'), Image.open(label_path).convert('L'), Image.open(ske_path).convert('L')
img_transform = transforms.Compose([
            tr.FixedResize(512),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()
        ])
image, gt, ske = img_transform((image, gt, ske))
print(image.shape, gt.shape, ske.shape)

image = image.unsqueeze(0).cuda()
# 定义钩子函数，用于获取模块的输出
finalconv_output = None
def hook(module, input, output):
    print("输出形状:", output.shape)
    print("输出值:", output)
    global finalconv_output
    finalconv_output = output
# 注册钩子函数到模型的某个模块
hook_handle = model.net_main.finalconv2.register_forward_hook(hook)

output, _, _ = model.net_main(image)
# preds = torch.argmax(output.data.cpu(), dim=1, keepdim=True)
# preds = preds.detach().cpu().numpy().astype(float)
# print(output[preds==1])
B,C,H,W = output.shape
# print(B, C, H, W)

feat_map_flat = output.permute(0, 2, 3, 1).reshape(-1, C)
mask_flat = gt.reshape(-1, 1)
print(feat_map_flat.shape, mask_flat.shape)
# 初始化t-SNE模型
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)

# 对特征进行降维
feat_map_flat = feat_map_flat.cpu().detach().numpy()
# x_min, x_max = np.min(feat_map_flat, 0), np.max(feat_map_flat, 0) #沿维度0求最值
# feat_map_flat = (feat_map_flat - x_min) / (x_max - x_min)
feat_tsne = tsne.fit_transform(feat_map_flat)
print(feat_tsne.shape)

label2color_dict = [[255, 165, 0], [0, 0, 255]]

feat_f = np.array([feat_tsne[i] for i in range(feat_tsne.shape[0]) if int(mask_flat[i]) == 1])
feat_b = np.array([feat_tsne[i] for i in range(feat_tsne.shape[0]) if int(mask_flat[i]) == 0])
print(len(feat_f), len(feat_b))
plt.figure(figsize=(8, 6))
color_f = label2color_dict[1]
color_f = [j / 255.0 for j in color_f]
plt.scatter(feat_f[:, 0], feat_f[:, 1], color=(color_f[0], color_f[1], color_f[2]), marker='.', s=1, alpha=0.5, label='road')
color_b = label2color_dict[0]
color_b = [j / 255.0 for j in color_b]
plt.scatter(feat_b[:, 0], feat_b[:, 1], color=(color_b[0], color_b[1], color_b[2]), marker='.', s=1, alpha=0.5, label='non-road')

# 隐藏边框和坐标轴
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
# for i in range(feat.shape[0]):
#     # label limitation
#     mask_i = int(mask_flat[i])
#     print(mask_i)
#     # plot
#     color = label2color_dict[mask_i]
#     color = [j / 255.0 for j in color]
#
#     plt.scatter(feat[i, 0], feat[i, 1], color=(color[0], color[1], color[2]), marker='.', linewidths=0.3)

# 可视化降维后的结果
plt.title('')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
plt.grid(False)
plt.legend()
plt.savefig('/home/arsc/tmp/pycharm_project_698/DA_Road/test_output/figures/final_after.png', dpi=500, bbox_inches='tight')#分辨率为 300 DPI，bbox_inches='tight'选项确保图形周围没有多余的空白。
plt.show()