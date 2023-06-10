import torch
import numpy as np
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
from torchvision import transforms
import datasets.custom_transforms as tr

class MyDataset(Dataset):
    def __init__(self,split='train'):
        self.split=split
        self.root_path = 'data/Massachusetts Roads Dataset'

        if self.split=='train':
            self.train_filenames=os.listdir(self.root_path+'/'+'crops/train_crops')
        elif self.split=='test':
            self.test_filenames=os.listdir(self.root_path+'/'+'crops/test_crops')
        else:
            self.val_filenames=os.listdir(self.root_path+'/'+'crops/val_crops')

    def __len__(self):
        if self.split=='train':
            return len(self.train_filenames)
        elif self.split=='test':
            return len(self.test_filenames)
        else:
            return len(self.val_filenames)

    def __getitem__(self, item):
        if self.split=='train':
            filename=self.train_filenames[item]
            image_path = self.root_path + '/crops/train_crops/'+filename
            label_path=self.root_path+'/crops/train_labels_crops/'+filename
        elif self.split=='test':
            filename=self.test_filenames[item]
            image_path = self.root_path + '/crops/test_crops/' + filename
            label_path = self.root_path + '/crops/test_labels_crops/' + filename
        else:
            filename=self.val_filenames[item]
            image_path = self.root_path + '/crops/val_crops/' + filename
            label_path = self.root_path + '/crops/val_labels_crops/' + filename

        img=Image.open(image_path).convert('RGB')#Image.open打开的为“BGR"格式
        label=Image.open(label_path)
        if self.split=='train':
            return self.transform_train(img,label)
        else:
            return self.transform_test(img,label)

        return (image,label)

    def transform_train(self,img,label):
        composed_transforms=transforms.Compose([
            tr.RandomRotate(180),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(512,512),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),#不同数据集不一样
            tr.ToTensor(),
        ])
        return composed_transforms((img,label))

    def transform_test(self,img,label):
        composed_transforms=transforms.Compose([
            tr.FixedResize(512),
            tr.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
            tr.ToTensor(),
        ])

        return composed_transforms((img,label))

