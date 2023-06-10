import torch
import numpy as np
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
from torchvision import transforms
import custom_transforms as ctr

class MyDataset(Dataset):
    def __init__(self,split='train'):
        self.split=split
        if self.split=='train':
            root_path = 'data/train'
            self.images_path=['data/train/images/'+path for path in os.listdir(os.path.join(root_path,'images'))]
        else:
            root_path='data/test'
            self.images_path=['data/test/images/'+path for path in os.listdir(os.path.join(root_path,'images'))]
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image_path=self.images_path[item]
        label_path='data/train/labels/'+image_path.split('/')[-1].split('sat')[0]+'mask.jpg'
        img=Image.open(image_path).convert('RGB')#Image.open打开的为“BGR"格式
        label=Image.open(label_path)
        if self.split=='train':
            return self.transform_train(img,label)
        else:
            return self.transform_test(img,label)

        return (image,label)

    def transform_train(self,img,label):
        composed_transforms=transforms.Compose([
            ctr.RandomRotate(180),
            ctr.RandomHorizontalFlip(),
            ctr.RandomScaleCrop(512,512),
            ctr.RandomGaussianBlur(),
            ctr.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
            ctr.ToTensor(),
        ])

        return composed_transforms((img,label))

    def transform_test(self,img,label):
        composed_transforms=transforms.Compose([
            ctr.FixedResize(512),
            ctr.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
            ctr.ToTensor(),
        ])

        return composed_transforms((img,label))
