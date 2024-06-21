import os
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import transforms
import numpy as np
import torch
import datasets.custom_transform as tr
from PIL import Image
from prefetch_generator import BackgroundGenerator
from itertools import cycle
from skimage.morphology import skeletonize

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ObjDataset(Dataset):
    def __init__(self, train_dir_root, train_images, image_size=512, mode='w'):
        self.train_dir_root=train_dir_root
        self.image_size = image_size
        self.train_images = train_images
        self.mode = mode
        # self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        self.img_transform_w = transforms.Compose([
            tr.RandomRotate(180),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(image_size, image_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # 不同数据集不一样
            tr.ToTensor()
        ])
        self.img_transform_s = transforms.Compose([
            tr.RandomRotate(180),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(image_size, image_size),
            # transforms.RandomAffine(degrees=90, translate=(0.5, 0.5), shear=30),
            # transforms.ColorJitter(hue=0.5),
            # transforms.RandomGrayscale(p=0.2),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # 不同数据集不一样
            tr.ToTensor()
        ])

    def __getitem__(self, index):
        image_path= self.train_dir_root+'/crops/images/'+ self.train_images[index]
        gt_path = self.train_dir_root + '/crops/gt/' + self.train_images[index]
        skeleton_path = self.train_dir_root + '/crops/skeleton/' + self.train_images[index]
        image = Image.open(image_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        skeleton = Image.open(skeleton_path).convert('L')
        if self.mode == 'w':
            image, gt, skeleton = self.img_transform_w((image, gt, skeleton))
        else:
            image, gt, skeleton = self.img_transform_s((image, gt, skeleton))
        return (image, gt, skeleton)

    def __len__(self):
        return len(self.train_images)


class ValObjDataset(Dataset):
    def __init__(self, val_dir_root, val_images, image_size):
        self.val_dir_root=val_dir_root
        self.val_images = val_images
        self.image_size = image_size
        self.img_transform = transforms.Compose([
            tr.FixedResize(image_size),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()
        ])

    def __getitem__(self, index):
        val_image_path=self.val_dir_root+'/crops/images/'+self.val_images[index]
        val_gt_path=self.val_dir_root+'/crops/gt/'+self.val_images[index]
        val_skeleton_path = self.val_dir_root + '/crops/skeleton/' + self.val_images[index]
        image = Image.open(val_image_path).convert('RGB')
        gt = Image.open(val_gt_path).convert('L')
        skeleton = Image.open(val_skeleton_path).convert('L')
        image, gt, skeleton = self.img_transform((image, gt, skeleton))

        return (image, gt, skeleton)

    def __len__(self):
        return len(self.val_images)

def deepglobe_dataset(dir_root, image_size=512, labeled_ratio=0.2):
    # train_images = [f for f in os.listdir(dir_root + '/images') if f.endswith('.jpg') or f.endswith('.png')]
    # val_images = [f for f in os.listdir(dir_root + '/images') if f.endswith('.jpg') or f.endswith('.png')]
    with open(dir_root + '/train_crops.txt') as ft:
        train_images = [f.strip() for f in ft.readlines()]
    with open(dir_root + '/val_crops.txt') as fv:
        val_images = [f.strip() for f in fv.readlines()]

    labeled_train_images = train_images[0:int(len(train_images) * labeled_ratio)]
    unlabeled_train_images = train_images[int(len(train_images) * labeled_ratio):]

    labeled_train_dataset = ObjDataset(dir_root, labeled_train_images, image_size, mode='w')
    unlabeled_train_dataset = ObjDataset(dir_root, unlabeled_train_images, image_size, mode='s')
    val_dataset = ValObjDataset(dir_root, val_images, image_size)

    return labeled_train_dataset, unlabeled_train_dataset, val_dataset

if __name__ =='__main__':
    deepglobe_dataset(dir_root='/home/arsc/tmp/pycharm_project_639/CSA/data/deepglobe/train/',image_size=512, labeled_ratio=0.2)#image为.jpg格式,gt为.png格式



