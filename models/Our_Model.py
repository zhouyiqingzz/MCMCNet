from utils.losses import MseLoss, EntropyMinimization, MixLoss2d, dice_bce_loss
from models.Contrastive_Loss import ContrastivePatchLoss, ContrastiveBank, ConLoss
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.DLinkNet5 import DLinkNet
import os
from torch.utils.checkpoint import checkpoint
import gc
gc.collect()

#可见设置，环境变量使得指定设备对CUDA应用可见
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1"
torch.backends.cudnn.enabled = True #寻找适合硬件的最佳算法,加上这一行运行时间能缩短很多!!!
torch.backends.cudnn.deterministic = True #由于计算中有随机性，每次网络前馈结果略有差异。设置该语句来避免这种结果波动
torch.backends.cudnn.benchmark = True #为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速

torch.cuda.empty_cache()  #释放显存
torch.cuda.empty_cache()  #释放显存
torch.cuda.empty_cache()  #释放显存

torch.manual_seed(1)

device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')

class DARNet(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.mode = opts.mode
        self.reduction = opts.reduction

        self.net_main = DLinkNet().cuda()
        self.net_ema = DLinkNet().cuda()

        self.sup_loss_seg = MixLoss2d().cuda()
        self.sup_loss_ske = MixLoss2d().cuda() #与下面输出通道为1时二选一
        # self.sup_loss_seg = dice_bce_loss().cuda()
        # self.sup_loss_ske = dice_bce_loss().cuda()
        self.conformity_loss_ske = MseLoss().cuda()

        self.contra_bank = ContrastiveBank().cuda()
        self.contra_patch_loss = ContrastivePatchLoss().cuda()

        self.entropy_minization = EntropyMinimization(self.reduction).cuda()
        self.contra_loss = ConLoss().cuda()


    def forward(self, labeled_img=None, labeled_seg_mask=None, labeled_ske_mask=None, unlabeled_img=None, mode='semi'):
        if mode == 'val':
            labeled_seg_pred, labeled_ske_pred, _ = self.net_main(labeled_img)
            return labeled_seg_pred
        else:
            labeled_seg_pred, labeled_ske_pred, _ = self.net_main(labeled_img) #此处注意修改labeled_project_main
            loss_sup_seg = self.sup_loss_seg(labeled_seg_mask.detach(), labeled_seg_pred)
            loss_sup_ske = self.sup_loss_ske(labeled_ske_mask.detach(), labeled_ske_pred)
            loss_sup = loss_sup_seg + loss_sup_ske #此处加权系数改动，请注意恢复 #道路骨架预测头消融实验位置

            unlabeled_seg_pred_main, unlabeled_ske_pred_main, project_output_main = self.net_main(unlabeled_img) # (2,512,512)
            unlabeled_seg_pred_ema, unlabeled_ske_pred_ema, project_output_ema = self.net_ema(unlabeled_img) # (2,512,512)
            loss_ent = self.entropy_minization(unlabeled_seg_pred_main)  # 防止过拟合或增加模型的稳定性
            # 消融实验之对比学习模块
            # loss_contra = self.contra_loss(project_output_main, project_output_ema) #该对损失为像素级对比损失，下述为patch级密集对比损失

            pseudo_label_seg = torch.argmax(unlabeled_seg_pred_ema.clone().detach(), dim=1, keepdim=True) #与下面输出通道为1时二选一
            pseudo_label_ske = torch.argmax(unlabeled_ske_pred_ema.clone().detach(), dim=1, keepdim=True)

            # patch级密集对比学习
            # ===============================
            loss_contra = 0.0
            label_ema = pseudo_label_seg.clone().float()
            label_main = torch.argmax(unlabeled_seg_pred_main.clone().detach(), dim=1, keepdim=True).float()
            neg_banks, pos_banks = self.contra_bank(project_output_main, project_output_ema, label_main, label_ema)
            if len(neg_banks) >= 35 and len(pos_banks) >= 35:
                # print(len(neg_banks), len(pos_banks))
                loss_contra = self.contra_patch_loss(project_output_main, project_output_ema, label_main, label_ema, neg_banks, pos_banks) #注意与上述一般对比损失二者选其一
            # ===============================

            loss_cons_seg = self.sup_loss_seg(pseudo_label_seg.detach(), unlabeled_seg_pred_main)
            loss_cons_ske = self.sup_loss_ske(pseudo_label_ske.detach(), unlabeled_ske_pred_main) #道路骨架预测头消融实验位置
            loss_cons = loss_cons_seg + 0.5 * loss_cons_ske  #道路骨架预测头消融实验位置，注意骨架损失的权重

            pseudo_label_seg2 = torch.argmax(unlabeled_seg_pred_main.clone(), dim=1, keepdim=True)
            pseudo_label_ske2 = torch.argmax(unlabeled_ske_pred_main.clone(), dim=1, keepdim=True) #与下面输出通道为1时二选一
            loss_conformity = self.conformity_loss_ske(pseudo_label_seg2[pseudo_label_ske2 == 1].float(), \
                                                           pseudo_label_ske2[pseudo_label_ske2 == 1].float()) + torch.tensor(1e-12)

            loss_total = loss_sup + loss_cons + 0.2 * loss_ent + 0.3 * loss_contra + 0.3 * loss_conformity # # 注意此处要加上对比损失
            return loss_total, 0.3 * loss_contra, 0.3 * loss_conformity

# def compute_contrastive(projector_output_main, projector_output_ema):
#     projector_output_main, projector_output_ema = projector_output_main.cuda(), projector_output_ema.cuda()
#     b, c, h, w = projector_output_main.shape #(4, 32, 64, 64)
#     # print(b,c,h,w)
#     patch_num = 8
#     hh, ww = h // patch_num, w // patch_num #(8,8)
#     features_patches_main = rearrange(projector_output_main, 'b c (p1 hh) (p2 ww) -> (b p1 p2 hh ww) c',
#                                      hh=h//patch_num, ww=w//patch_num, p1=patch_num, p2=patch_num, c=c) #(4, 16, 128, 128)-> (4*8*8*16*16,16)
#     features_patches_ema = rearrange(projector_output_ema, 'b c (p1 hh) (p2 ww) -> (b p1 p2 hh ww) c',
#                                      hh=h//patch_num, ww=w//patch_num, p1=patch_num, p2=patch_num, c=c) #(4, 16, 128, 128)-> (4*8*8*16*16,16)
#     features_patches_total = torch.cat([features_patches_main, features_patches_ema], dim=0) #(32768, 16)
#
#     loss_contr_sum = 0.0
#     loss_contr_count =0
#     # print(features_patches_main.shape, features_patches_ema.shape)
#     constrastive_loss = ContrastivePatchLoss().cuda()
#     for i_patch in range(b * patch_num * patch_num):
#         patch_main_i = features_patches_main[i_patch * hh * ww : (i_patch +1) * hh * ww , :] # hh*ww, c = 8*8, 128
#         patch_ema_i = features_patches_ema[i_patch * hh * ww : (i_patch +1) * hh * ww , :] # hh*ww, c = 8*8, 128
#
#         loss_contr = checkpoint(constrastive_loss, patch_main_i, patch_ema_i, features_patches_total)
#         loss_contr = loss_contr.sum()
#         loss_contr_sum += loss_contr
#         loss_contr_count += 1
#
#     loss_contr_final = loss_contr_sum / loss_contr_count #出现除数为0的情况会出现inf，NaN表示属于浮点型但不是一个数字
#     # print('loss_contr_final:{}'.format(loss_contr_final))
#
#     return loss_contr_final
