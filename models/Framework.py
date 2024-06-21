import torch
# from torch._C import long
import torch.nn as nn
from torch.autograd import Variable as V
from discriminator import FCDiscriminator
from dis_loss import bce_loss
from mse_loss import mse_loss
from scipy.ndimage.morphology import *

import cv2
import numpy as np
from skimage.morphology import skeletonize


class MyFrame():
    def __init__(self, net, loss, lr=2e-4, mode='test', evalmode=False):
        self.net = net().cuda()
        self.model_D1 = FCDiscriminator(num_classes=512)

        self.model_D1.train()
        self.model_D1.cuda()
        # self.net = nn.parallel.DistributedDataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.optimizer_D1 = torch.optim.Adam(self.model_D1.parameters(), lr=1e-4, betas=(0.9, 0.99))

        self.loss = loss()
        self.loss1 = bce_loss()
        self.loss2 = mse_loss()

        self.old_lr = lr
        self.mode = mode
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, labeled_img=None, labeled_seg_mask=None, labeled_skeleton_mask=None, unlabeled_img=None, unlabeled_seg_mask=None, unlabeled_skeleton_mask=None,
                  img_id=None):  # t_img_batch, dist_mask=None,  t_img_batch, t_ske_mask=None, ske_mask = None,
        self.labeled_img = labeled_img
        self.unlabeled_img = unlabeled_img
        self.labeled_seg_mask = labeled_seg_mask
        self.unlabeled_seg_mask = unlabeled_seg_mask
        self.labeled_skeleton_mask = labeled_skeleton_mask
        self.unlabeled_skeleton_mask = unlabeled_skeleton_mask

    def forward(self, volatile=False):
        if self.mode == 'test':
            self.img = V(self.img.cuda())
            if self.mask is not None:
                self.mask = V(self.mask.cuda())
        else:
            self.labeled_img = V(self.labeled_img.cuda())
            self.unlabeled_img = V(self.unlabeled_img.cuda())
            if self.mask is not None:
                self.labeled_seg_mask = V(self.labeled_seg_mask.cuda())
                self.unlabeled_seg_mask = V(self.unlabeled_seg_mask.cuda())
                self.labeled_skeleton_mask = V(self.labeled_skeleton_mask.cuda())
                self.unlabeled_skeleton_mask = V(self.unlabeled_skeleton_mask.cuda())

    def optimize(self):
        ct_loss_ = True
        adv_loss_ = True

        self.optimizer.zero_grad()

        self.img_c = torch.cat((self.labeled_img, self.unlabeled_img))
        self.seg_mask_c = torch.cat((self.labeled_seg_mask, self.unlabeled_seg_mask))
        self.skeleton_mask_c = torch.cat((self.labeled_skeleton_mask, self.unlabeled_skeleton_mask))

        pred_seg, pred_skeleton, e4 = self.net.forward(self.img_c)

        loss_seg = self.loss(self.seg_mask_c, pred_seg)
        loss_skeleton = self.loss(self.skeleton_mask_c, pred_skeleton)

        loss_road = loss_seg + loss_skeleton
        if not torch.isnan(loss_road):
            loss_road.backward(retain_graph=True)
        else:
            print('here')

        if ct_loss_:
            pred_seg = torch.sigmoid(pred_seg)
            pred_skeleton = torch.sigmoid(pred_skeleton)

            loss_conformity = 0.001 * self.loss2(pred_skeleton[self.skeleton_mask_c == 1], pred_seg[self.skeleton_mask_c == 1]) + torch.tensor(1e-12)

            if not (torch.isnan(loss_conformity) or loss_conformity == torch.tensor(1e-12)):
                if adv_loss_:
                    loss_conformity.backward(retain_graph=True)
                else:
                    loss_conformity.backward()
            else:
                print('here')

        if adv_loss_:
            self.optimizer_D1.zero_grad()
            for param in self.model_D1.parameters():
                param.requires_grad = False

            source_label = 0
            target_label = 1

            mid = int(e4.shape[0] / 2)
            D_out1 = self.model_D1(e4[mid:, :, :, :])
            loss_adv_target1 = self.loss1(D_out1, V(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
            loss_adv_target1 = 0.01 * loss_adv_target1
            loss_adv_target1.backward()

            self.optimizer.step()

            for param in self.model_D1.parameters():
                param.requires_grad = True

            # train D with source
            f_pred1 = e4.detach()
            D_out1_1 = self.model_D1(f_pred1[:mid, :, :, :])
            loss_D1 = self.loss1(D_out1_1, V(torch.FloatTensor(D_out1_1.data.size()).fill_(source_label)).cuda())
            loss_D1.backward(retain_graph=True)

            pred_target1 = f_pred1[mid:, :, :, :]
            D_out1_2 = self.model_D1.forward(pred_target1)
            loss_D2 = self.loss1(D_out1_2, V(torch.FloatTensor(D_out1_2.data.size()).fill_(target_label)).cuda())
            loss_D2.backward()

            self.optimizer_D1.step()
        else:
            self.optimizer.step()

        lost_total = loss_road.item()
        return lost_total, loss_adv_target1.item(), loss_conformity.item()

    def save(self, path, path1):
        torch.save(self.net.state_dict(), path)
        torch.save(self.model_D1.state_dict(), path1)
