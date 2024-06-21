import torch
import numpy as np
import torch.nn as nn
from PIL import Image,ImageOps,ImageFilter
import random
from skimage import morphology
random.seed(1)
#image size:(256,256)
class Normalize(object):
    """
    Normalize a tensor image with mean and std
    """
    def __init__(self,mean=(0.429,0.432,0.396),std=(0.174,0.169,0.172)):
        self.mean=mean
        self.std=std

    def __call__(self,img_label):
        img, label, skeleton = img_label
        # print(img.size)
        img=np.array(img).astype(np.float32)
        label=np.array(label).astype(np.float32)
        skeleton = np.array(skeleton).astype(np.float32)

        img /= 255
        img -= self.mean
        img /= self.std

        label /= 255
        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        skeleton /= 255
        skeleton[skeleton >= 0.5] = 1
        skeleton[skeleton < 0.5] = 0

        return (img,label,skeleton)

class RandomHorizontalFlip(object):
    def __call__(self,img_label):
        img, label, skeleton = img_label
        # print(img.size)

        if random.random()<0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            skeleton = skeleton.transpose(Image.FLIP_LEFT_RIGHT)
        return (img, label, skeleton)

class RandomRotate(object):
    def __init__(self,degree):
        self.degree=degree

    def __call__(self,img_label):
        img, label, skeleton = img_label

        rotate_degree=random.uniform(-1*self.degree,self.degree)
        img=img.rotate(rotate_degree,Image.BILINEAR)
        label=label.rotate(rotate_degree,Image.BILINEAR)
        skeleton = skeleton.rotate(rotate_degree, Image.BILINEAR)

        return (img,label,skeleton)

class RandomGaussianBlur(object):
    def __call__(self,img_label):
        img, label, skeleton = img_label

        if random.random()<0.5:
            img=img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return (img,label,skeleton)

class RandomScaleCrop(object):
    def __init__(self,base_size,crop_size,fill=0):
        self.base_size=base_size
        self.crop_size=crop_size
        self.fill=fill

    def __call__(self,img_label):
        img, label,skeleton = img_label
        # print(img.size)

        short_size=random.randint(int(self.base_size*0.5),int(self.base_size*2.0))
        w,h=img.size
        if h>w:
            ow=short_size
            oh=int(1.0*h*ow/w)
        else:
            oh=short_size
            ow=int(1.0*w*oh/h)
        img = img.resize((ow,oh),Image.BILINEAR)
        label = label.resize((ow,oh),Image.NEAREST)
        skeleton = skeleton.resize((ow, oh), Image.NEAREST)

        if short_size<self.crop_size:
            padh=self.crop_size-oh if oh<self.crop_size else 0
            padw=self.crop_size-ow if ow<self.crop_size else 0
            img=ImageOps.expand(img,border=(0,0,padw,padh),fill=self.fill)
            label=ImageOps.expand(label,border=(0,0,padw,padh),fill=self.fill)
            skeleton = ImageOps.expand(skeleton, border=(0, 0, padw, padh), fill=self.fill)

        w,h=img.size
        x1=random.randint(0,w-self.crop_size)
        y1=random.randint(0,h-self.crop_size)
        img=img.crop((x1,y1,x1+self.crop_size,y1+self.crop_size))
        label=label.crop((x1,y1,x1+self.crop_size,y1+self.crop_size))
        skeleton = skeleton.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return (img,label,skeleton)

class FixScaleCrop(object):
    def __init__(self,crop_size):
        self.crop_size=crop_size

    def __call__(self,img_label):
        img, label, skeleton = img_label
        w,h=img.size
        if w>h:
            oh=self.crop_size
            ow=int(1.0*w*oh/h)
        else:
            ow=self.crop_size
            oh=int(1.0*h*ow/w)
        img=img.resize((ow,oh),Image.BILINEAR)
        label=label.resize((ow,oh),Image.NEAREST)
        skeleton = skeleton.resize((ow, oh), Image.NEAREST)

        w,h=img.size
        x1=int(round(w-self.crop_size)/2.0)
        y1=int(round(h-self.crop_size)/2.0)
        img=img.crop((x1,y1,x1+self.crop_size,y1+self.crop_size))
        label=label.crop((x1,y1,x1+self.crop_size,y1+self.crop_size))
        skeleton = skeleton.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return (img,label,skeleton)

class FixedResize(object):
    def __init__(self,size):
        self.size=(size,size)

    def __call__(self,img_label):
        img, label, skeleton = img_label
        img=img.resize(self.size,Image.BILINEAR)
        label=label.resize(self.size,Image.NEAREST)
        skeleton = skeleton.resize(self.size, Image.NEAREST)

        return (img,label,skeleton)

class ToTensor(object):
    def __call__(self,img_label):
        img, label, skeleton = img_label
        img = np.array(img).astype(np.float32).transpose((2,0,1))
        label = np.array(label).astype(np.float32)
        skeleton = np.array(skeleton).astype(np.float32)
        img=torch.from_numpy(img).float()
        label=torch.from_numpy(label).float()
        skeleton = torch.from_numpy(skeleton).float()

        return (img,label,skeleton)