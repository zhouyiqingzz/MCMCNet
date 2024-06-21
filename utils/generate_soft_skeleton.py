import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter=40):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, img):
        return self.soft_skel(img)

if __name__=='__main__':
    skeletonize = SoftSkeletonize()
    # x = torch.randn(1, 3, 128, 128)
    # output = skeletonize(x)
    # print(output.shape)

    label_path = '/home/arsc/tmp/pycharm_project_503/CDCL/data/Massachusetts/crops/val_labels_crops/10228690_15_0_0.jpg'
    label = Image.open(label_path).convert('L')  # 替换为您的图像路径
    label = np.array(label).astype(np.float32)  # 将图像转换为 0-255 的灰度图像
    label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
    print(label.shape)
    skeleton_label = skeletonize(label)

    skeleton_label = skeleton_label.squeeze(0).squeeze(0)
    skeleton_label = np.array(skeleton_label).astype(np.uint8)
    skeleton_label = Image.fromarray(skeleton_label)
    skeleton_label.save('aa.jpg')