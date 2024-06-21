import torch
import torchvision
import torch.nn as nn
import os
import cv2
import torch.nn.functional as F
import numpy as np
from models.Our_Model import DARNet
from utils.options import Opts
from PIL import Image
import requests
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from pytorch_grad_cam import GradCAM
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class SemanticSegmentationTarget():
    def __init__(self,category,mask):
        self.category=category
        self.mask=torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask=self.mask.cuda()

    def __call__(self,model_output):#要将主输出和辅助头输出区分开，重点提取出主输出
        return (model_output[0][self.category,:,:]*self.mask).sum()

#定义模型包装器来获取输出张量，因为pytorch模型输出一个自定义字典
class SegmentationModelOutputWrapper(nn.Module):
    def __init__(self,model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model=model
    def forward(self,x):
        return self.model(x)

def generate_heatmap(parent_path, img_name, save_path, model_path):
    img_path=os.path.join(parent_path, img_name)
    image = Image.open(img_path)
    rgb_image = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).cuda()#.cpu()
    opts = Opts()
    model=DARNet(opts)#.cuda()
    model=model.eval()
    checkpoint=torch.load(model_path, map_location='cuda')
    # print(checkpoint['state_dict'])
    for key,param in list(checkpoint['state_dict'].items()):
        if key.startswith('module.'):
            checkpoint['state_dict'][key[7:]]=param
            checkpoint['state_dict'].pop(key)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    model = SegmentationModelOutputWrapper(model.net_main)
    output, _, _ = model(input_tensor)

    normalized_masks = F.softmax(output, dim=1)#.cpu()#四维
    sem_classes = [
        '__background__', 'road'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    road_category = sem_class_to_idx['road']#1
    road_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()#三维
    # print(road_mask[road_mask==1])
    road_mask_uint8 = 255 * np.uint8(road_mask == road_category)
    # road_mask_float = np.float32(road_mask == road_category)

    #查看某一特征层的输出
    target_layers=[model.model.finalconv3] #必须具体到卷积层，此处得到的特征图也可以用于t-SNE中的可视化
    targets=[SemanticSegmentationTarget(road_category, road_mask_uint8)]
    print(targets)
    with GradCAM(model=model,target_layers=target_layers) as cam:
        grayscale_cam=cam(input_tensor=input_tensor,targets=targets)[0,:]
        cam_image=show_cam_on_image(rgb_image,grayscale_cam,use_rgb=True)
        print(cam_image.shape)
        img_new=Image.fromarray(cam_image)
        img_new.save(os.path.join(save_path, img_name))

if __name__=='__main__':
    CHN6_parent_path = '/home/arsc/tmp/pycharm_project_639/CSA/data/CHN6/val/images'
    CHN6_img_names = ['bj100528_sat.jpg']#os.listdir(CHN6_parent_path)
    # CHN6_img_names=['am100761_sat.jpg', 'am100762_sat.jpg', 'am100763_sat.jpg', 'am100764_sat.jpg', 'am100765_sat.jpg', 'am100766_sat.jpg']
    # CHN6_img_names = ['bj100539_sat.jpg', 'bj100529_sat.jpg', 'bj100478_sat.jpg']
    # Massachu_parent_path = '/home/arsc/tmp/pycharm_project_503/CDCL/data/Massachusetts/crops/val_crops'
    # Massachu_img_names=['10228690_15_0_0.jpg', '10228690_15_0_1.jpg', '10228690_15_0_2.jpg', '10228690_15_1_0.jpg']
    # Massachu_img_names = os.listdir(Massachu_parent_path)
    DeepGlobe_parent_path = '/home/arsc/tmp/pycharm_project_639/CSA/data/deepglobe/train/crops/images'
    DeepGlobe_img_names = ['9339_0_0.png', '9339_0_1.png', '9339_1_0.png', '9339_1_1.png']

    out_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    CHN6_save_path='/home/arsc/tmp/pycharm_project_698/DA_Road/test_output/heatmap/CHN6/' + out_id
    # Massachu_save_path = '/home/arsc/tmp/pycharm_project_639/CSA/test_output/heatmap/Massachusetts/' + out_id
    DeepGlobe_save_path = '/home/arsc/tmp/pycharm_project_698/DA_Road/test_output/heatmap/DeepGlobe/' + out_id

    if not os.path.exists(CHN6_save_path):
        try:
            os.mkdir(CHN6_save_path)
        except FileExistsError:
            pass
    # if not os.path.exists(Massachu_save_path):
    #     try:
    #         os.mkdir(Massachu_save_path)
    #     except FileExistsError:
    #         pass
    if not os.path.exists(DeepGlobe_save_path):
        try:
            os.mkdir(DeepGlobe_save_path)
        except FileExistsError:
            pass
    CHN6_model_path='/home/arsc/tmp/pycharm_project_698/DA_Road/run_model_experiments/our_model/CHN6/experiment_20240409_104239/20_checkpoint.pth'
    # CHN6_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/CHN6/experiment_20240131_184053/5_checkpoint.pth'
    # CHN6_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/CHN6/experiment_20240131_030109/94_checkpoint.pth'
    # CHN6_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/CHN6/experiment_20240131_134731/20_checkpoint.pth' #==>best
    # CHN6_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/CHN6/experiment_20240202_055734/18_checkpoint.pth'
    # CHN6_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/CHN6/experiment_20240201_052746/0_checkpoint.pth'
    # CHN6_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/CHN6/experiment_20240201_063133/8_checkpoint.pth'
    # Massachu_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/Massachusetts/experiment_20231230_084959/110_checkpoint.pth'
    # Massachu_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/Massachusetts/experiment_20231230_084959/100_checkpoint.pth'
    # DeepGlobe_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/DeepGlobe/experiment_20231231_174709/106_checkpoint.pth'
    # DeepGlobe_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/DeepGlobe/experiment_20231231_174709/100_checkpoint.pth'
    # DeepGlobe_model_path='/home/arsc/tmp/pycharm_project_639/CSA/run_Model_experiments/OurModel/DeepGlobe/experiment_20230912_120139/145_checkpoint.pth'
    # DeepGlobe_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/DeepGlobe/experiment_20240202_090101/10_checkpoint.pth'
    # DeepGlobe_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/DeepGlobe/experiment_20240202_154948/5_checkpoint.pth'
    # DeepGlobe_model_path='/home/arsc/tmp/pycharm_project_399/CSA/run_model_experiments/OurModel/DeepGlobe/experiment_20240202_154948/35_checkpoint.pth'
    # DeepGlobe_model_path='/home/arsc/tmp/pycharm_project_698/DA_Road/run_model_experiments/our_model/DeepGlobe/experiment_20240409_013435/31_checkpoint.pth'
    for img_name in CHN6_img_names:
        generate_heatmap(CHN6_parent_path, img_name, CHN6_save_path, CHN6_model_path)
    # for img_name in Massachu_img_names:
    #     generate_heatmap(Massachu_parent_path, img_name, Massachu_save_path, Massachu_model_path)
    # for img_name in DeepGlobe_img_names:
    #     generate_heatmap(DeepGlobe_parent_path, img_name, DeepGlobe_save_path, DeepGlobe_model_path)

