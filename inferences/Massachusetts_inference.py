import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from datasets import custom_transform as tr
from utils.metrics import Evaluator
import os
from models.Our_Model import DARNet
from datasets.Massachusetts_dataset import ObjDataset
from utils.options import Opts
import warnings
import torch.distributed as dist
from datetime import datetime
import argparse
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1"
torch.backends.cudnn.enabled =False #寻找适合硬件的最佳算法
torch.backends.cudnn.deterministic = True #由于计算中有随机性，每次网络前馈结果略有差异。设置该语句来避免这种结果波动
torch.backends.cudnn.benchmark = True #为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速

def transform_test(sample):
    composed_transforms = transforms.Compose([
        tr.FixedResize(size=512),
        tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        tr.ToTensor()
    ])
    return composed_transforms(sample)

def inference_result():
    #模型加载
    opts = Opts()
    model=DARNet(opts)#修改(1)
    model.eval()
    # model=model.cuda()
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)

    model_path = '/home/arsc/tmp/pycharm_project_698/DA_Road/run_model_experiments/our_model/Massachusetts/experiment_20240302_083630/184_checkpoint.pth'#修改(2)
    ckpt=torch.load(model_path,map_location='cpu')
    for key, param in list(ckpt['state_dict'].items()):
        if key.startswith('module.'):
            ckpt['state_dict'][key[7:]] = param
            ckpt['state_dict'].pop(key)
    model.load_state_dict(ckpt['state_dict'],strict=True)
    #确定输出路径
    out_id=datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = '../test_output/Massachusetts/OurModel/'  #+ out_id #修改(3)
    if not os.path.exists(out_path):
        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass
    base_dir='/home/arsc/tmp/pycharm_project_639/CSA/data/Massachusetts/crops'
    test_imgnames=os.listdir(os.path.join(base_dir,'val_crops'))
    #加载测试集并测试
    count=0
    for i, imgname in enumerate(test_imgnames):
        count+=1
        imgname=imgname.strip()
        img_path=os.path.join(base_dir+'/val_crops',imgname)
        label_path=os.path.join(base_dir+'/val_labels_crops',imgname)
        skeleton_path = os.path.join(base_dir + '/val_skeletons_crops', imgname)
        ori_img = Image.open(img_path).convert('RGB')
        ori_label = Image.open(label_path).convert('L')
        ori_skeleton = Image.open(skeleton_path).convert('L')
        imgs,labels,skeletons=transform_test((ori_img,ori_label,ori_skeleton))

        imgs, labels, skeletons = imgs.unsqueeze(0),labels.unsqueeze(0), skeletons.unsqueeze(0)
        imgs, labels, skeletons = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True), skeletons.cuda(
            non_blocking=True)
        with torch.no_grad():
            loss_total, labeled_seg_pred, labeled_skeleton_pred = model(labeled_img=imgs, labeled_seg_mask=labels,
                                                                             labeled_skeleton_mask=skeletons,
                                                                             unlabeled_img=None, epoch=None, iter=None,
                                                                             num_iters=None, mode='sup_only')
        # 输出通道是2时
        preds = torch.argmax(labeled_seg_pred.data.cpu(), dim=1, keepdim=True)
        preds = preds.detach().cpu().numpy().astype(float).squeeze(0).squeeze(0)

        preds[preds>0]=255
        pred = Image.fromarray(preds).convert('L')
        pred.save(os.path.join(out_path, imgname))

if __name__=='__main__':
    inference_result()