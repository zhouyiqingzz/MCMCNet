# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:37:12 2022

@author: loua2
"""
import torch
from torch import nn
import torch.nn.functional as F
import warnings
from torch.autograd import Variable
from utils.losses import FocalLoss2d
torch.manual_seed(1)
warnings.filterwarnings("ignore")
# torch.autograd.set_detect_anomaly(True)#异常调试，实际运行时要设置为False，否则会费时

device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')

class ConLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, base_temperature=0.5):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss2d().cuda()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        #feat_q, feat_k = feat_q.cuda(), feat_k.cuda()
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1) #(B, hw, c)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1) #(B, hw, c)
        feat_q = F.normalize(feat_q.clone(), dim=-1, p=1)
        feat_k = F.normalize(feat_k.clone(), dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1)) #(Bhw,1,c)*(Bhw,c,1)->(Bhw,1,1)
        l_pos = l_pos.view(-1, 1) #(Bhw,1)

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)#(B,hw,c)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)#(B,hw,c)
        npatches = feat_q.size(1)#hw
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))#(B,hw,c)*(B,c,hw)->(B,hw,hw)

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :] #(1,hw,hw)

        l_neg_curbatch.masked_fill_(diagonal, -100.0) #对角线位置填充为-10.0
        l_neg = l_neg_curbatch.view(-1, npatches)#(Bhw,hw)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature#(Bhw,1),(Bhw,hw)->(Bhw,1+hw)
        temp = out.size(0)#Bhw
        # loss = self.cross_entropy_loss(out, torch.zeros(temp, dtype=torch.long,device=feat_q.device))#(Bhw,hw+1),(Bhw), #交叉熵损失看的是概率
        loss = self.focal_loss(torch.zeros(temp, dtype=torch.long, device=feat_q.device), out)

        return loss


class contrastive_loss_sup(torch.nn.Module):
    def __init__(self, temperature=0.2, base_temperature=0.2):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(contrastive_loss_sup, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        #feat_q, feat_k = feat_q.cuda(), feat_k.cuda()
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q.clone(), dim=-1, p=1)
        feat_k = F.normalize(feat_k.clone(), dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.zeros((batch_size * width * width, 1)).cuda() #需要修改，请注意 #(Bhw,1)
        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -100.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature
        temp = out.size(0)
        loss = self.cross_entropy_loss(out, torch.zeros(temp, dtype=torch.long,device=feat_q.device))

        return loss


# class ContrastivePatchLoss(nn.Module):
#     def __init__(self, bdp_threshold=0.2, fdp_threshold=0.7, temp=0.1, eps=1e-8):
#         super().__init__()
#         self.temp = temp
#         self.eps = eps
#         self.bdp_threshold = bdp_threshold
#         self.fdp_threshold = fdp_threshold
#
#     def forward(self, anchor, pos_pair, neg_pair): #(64,128),(64,128), (32768,128), (64,1),(32768,1)
#         pos = torch.div(torch.mul(anchor, pos_pair), self.temp).sum(-1, keepdim=True) #(64,128)*(128,64)->(64,64) ->(64,1)
#
#         neg = torch.div(torch.matmul(anchor, neg_pair.T), self.temp) #(64,128)*(128, 32768)->(64, 32768)
#         neg = torch.cat([pos, neg], 1) #(64, 1+32768)->(64, 32769)
#         max = torch.max(neg, dim=1, keepdim=True)[0] #(64, 1) #keepdim=True表示输出与输入维度相同，此例中都是二维
#         exp_neg = (torch.exp(neg - max)).sum(-1) #(64, 32769)->(64,),此时sum(-1)中默认keepdim=False即减少一个维度从二维变为一维
#
#         loss = torch.exp(pos - max).squeeze(-1) / (exp_neg + self.eps) #(64,)/(64,) ->(64,)
#         loss = -torch.log(loss + self.eps) #(64,)
#         return loss

class ContrastiveBank(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.pos_banks = []
        self.neg_banks = []
        self.patch_num = 8
        self.N = 35

    def forward(self, main_out, ema_out, main_label, ema_label):
        B, C, H, W = main_out.shape
        h, w = H // self.patch_num, W // self.patch_num
        hh, ww = 4 * h, 4 * w
        for i in range(self.patch_num * self.patch_num):
            j = i // self.patch_num
            k = i % self.patch_num
            for ii in range(B):
                main_out_patch = main_out[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]
                ema_out_patch = ema_out[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]
                main_label_patch = main_label[ii, j * hh: (j + 1) * hh, k * ww: (k + 1) * ww]
                ema_label_patch = ema_label[ii, j * hh: (j + 1) * hh, k * ww: (k + 1) * ww]

                if torch.mean(main_label_patch) < 0.1:
                    self.neg_banks.append(main_out_patch)
                    if len(self.neg_banks) > self.N:
                        self.neg_banks.pop(0)
                elif torch.mean(main_label_patch) >= 0.1:
                    self.pos_banks.append(main_out_patch)
                    if len(self.pos_banks) > self.N:
                        self.pos_banks.pop(0)
                if torch.mean(ema_label_patch) < 0.1:
                    self.neg_banks.append(ema_out_patch)
                    if len(self.neg_banks) > self.N:
                        self.neg_banks.pop(0)
                elif torch.mean(ema_label_patch) >= 0.1:
                    self.pos_banks.append(ema_out_patch)
                    if len(self.pos_banks) > self.N:
                        self.pos_banks.pop(0)
        return self.neg_banks, self.pos_banks


class ContrastivePatchLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.temp = 0.5
        self.patch_num = 8
        self.eps = 1e-5

    def forward(self, main_out, ema_out, main_label, ema_label, neg_banks, pos_banks):
        B, C, H, W = main_out.shape
        h, w = H // self.patch_num, W // self.patch_num
        neg_len, pos_len = len(neg_banks), len(pos_banks)
        # print(neg_len, pos_len)
        hh, ww = 4 * h, 4 * w
        neg_banks = torch.tensor([item.cpu().detach().numpy() for item in neg_banks]).permute(0, 2, 3, 1).reshape(neg_len * h * w, C).cuda()
        pos_banks = torch.tensor([item.cpu().detach().numpy() for item in pos_banks]).permute(0, 2, 3, 1).reshape(pos_len * h * w, C).cuda()
        loss = 0.0
        for i in range(self.patch_num * self.patch_num):
            j = i // self.patch_num
            k = i % self.patch_num
            for ii in range(B):
                main_out_patch = main_out[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]
                ema_out_patch = ema_out[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]
                main_label_patch = main_label[ii, j * hh: (j + 1) * hh, k * ww: (k + 1) * ww]
                # ema_label_patch = ema_label[ii, j * hh: (j + 1) * hh, k * ww: (k + 1) * ww]

                anchor = main_out_patch.permute(1, 2, 0).reshape(h * w, C).cuda()
                if torch.mean(main_label_patch) < 0.1:
                    neg_sim = torch.div(torch.matmul(anchor, pos_banks.T), self.temp) # h*w, 100*h*w
                elif torch.mean(main_label_patch) >= 0.1:
                    neg_sim = torch.div(torch.matmul(anchor, neg_banks.T), self.temp) # h*w, 100*h*w
                pos_pair = ema_out_patch.permute(1, 2, 0).reshape(h * w, C).cuda()
                pos_sim = torch.div(torch.mul(anchor, pos_pair), self.temp).sum(-1, keepdim=True) #  h*w, 1

                neg = torch.cat([pos_sim, neg_sim], 1)
                max = torch.max(neg, dim=1, keepdim=True)[0]
                exp_neg = (torch.exp(neg - max)).sum(-1) # h*w

                loss_patch = torch.exp(pos_sim - max).squeeze(-1) / (exp_neg + self.eps) # h*w
                loss_patch = -torch.log(loss_patch + self.eps) # h*w
                loss += loss_patch.mean()

        loss = loss / (B * self.patch_num * self.patch_num)
        return loss



if __name__=='__main__':
    pixel_wise_contrastive_loss_criter = ConLoss().cuda()
    contrastive_loss_sup_criter = contrastive_loss_sup().cuda()
    feat_q=torch.randn((1,16,128,128)).cuda()
    feat_k = torch.randn((1, 16, 128, 128)).cuda()
    feat_l_q=torch.randn((1,32,64,64)).cuda()
    feat_l_k = torch.randn((1, 32, 64, 64)).cuda()
    Loss_contrast = Variable(pixel_wise_contrastive_loss_criter(feat_q, feat_k))
    Loss_contrast_2 = Variable(contrastive_loss_sup_criter(feat_l_q, feat_l_k))
    # Loss_contrast.backward()
    # Loss_contrast_2.backward()
    print(Loss_contrast,Loss_contrast_2)



