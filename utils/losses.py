import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class MseLoss(nn.Module):#均方误差损失
    def __init__(self, batch=True):
        super().__init__()
        self.batch = batch
        self.mse_loss = nn.MSELoss()  # reduction='none'

    def __call__(self, y_pred, y_true):
        if y_true.sum() == 0:
            a = 0.0
        else:
            a = self.mse_loss(y_pred, y_true)
        return a

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()  # 需配合nn.Sigmoid来使用
        self.focalLoss2d = FocalLoss2d()

    def soft_dice_coeff(self, y_true, y_pred):  # y_true:(b,c,h,w),y_pred:(b,c,h,w)(c值为1)
        smooth = 1e-8
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2.0 * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1.0 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    # def prospect_entropy_loss(self, target, pred):  # 关注前景道路的损失#y_true:(b,c,h,w),y_pred:(b,c,h,w)(c值为1)
    #     b, c, h, w = pred.size()
    #     ignore_label = 0
    #     smooth = 1e-8
    #     target = target.view(-1)  # 变乘1维
    #     pred = pred.view(-1)
    #     valid_mask = target.ne(ignore_label)  # ne:不等于
    #     target = target * valid_mask.long()  # 突出有效部分，无效部分(ignore_label)被遮挡住
    #     pred = pred * valid_mask.long()
    #     num_valid = valid_mask.sum()  # 有效元素的总和 #torch.sum(keepdim=True)保证相加后张量维度不会被压缩
    #     oup = -1.0 / (num_valid + smooth)  # torch.sum()（或用某某tensor张量.sum()）计算整个矩阵和
    #     # print(target.shape,pred.shape,oup.shape)#扩展新维度新的维度必须在最前面
    #
    #     loss = (target * torch.log(pred + smooth) * oup).sum()
    #     # print(loss)
    #     return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return 0.7 * a + 0.3 * b  # +0.2*c


class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction =reduction

    def forward(self, inputs, targets, mask=None):
        _targets = targets.clone().long()
        if mask is not None:
            _targets[mask] = self.ignore_index
        loss = F.cross_entropy(inputs, _targets)
        return loss

class DiceLossMulticlass(torch.nn.Module):
    def __init__(self, num_classes):
        super(DiceLossMulticlass, self).__init__()
        self.num_classes = num_classes

    def forward(self, target, input):  # input:(B,C,H,W),target:(B,H,W)
        smooth = 1e-8  # 平滑项，避免分母为零
        loss = 0
        input = nn.Softmax(dim=1)(input)
        for class_idx in range(self.num_classes):
            input_class = input[:, class_idx, :, :]
            target_class = (target == class_idx).float()

            intersection = torch.sum(input_class * target_class)
            union = torch.sum(input_class) + torch.sum(target_class)
            dice_score = (2. * intersection + smooth) / (union + smooth)
            class_loss = 1. - dice_score

            loss += class_loss
        return loss / self.num_classes

class FocalLoss2d(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss2d, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,target, input):
        ce_loss = F.cross_entropy(input, target.long(), reduction='none') #要保证target为整型
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MixLoss2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLossMulticlass(2)
        self.focal_loss = FocalLoss2d()

    def __call__(self, targets, preds):
        targets=targets.squeeze(1)
        # print(preds.shape, targets.shape)
        a = self.dice_loss(targets, preds)
        b = self.focal_loss(targets, preds)

        return 0.3*a + 0.7*b

class EntropyMinimization(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs):
        P = torch.softmax(inputs, dim=1)
        logP = torch.log_softmax(inputs, dim=1)
        PlogP = P * logP
        loss_ent = -1.0 * PlogP.sum(dim=1)
        if self.reduction == 'mean':
            loss_ent = torch.mean(loss_ent)
        elif self.reduction == 'sum':
            loss_ent = torch.sum(loss_ent)
        else:
            pass
        return loss_ent

class ContrastiveLoss(nn.Module):
    def __init__(self, bdp_threshold, fdp_threshold, temp=0.1, eps=1e-8):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.eps = eps
        self.bdp_threshold = bdp_threshold
        self.fdp_threshold = fdp_threshold

    def forward(self, anchor, pos_pair, neg_pair, pseudo_label_1, pseudo_label_2, pseudo_label_all, fcs_1, fcs_2, fcs_all): #(64,128),(64,128), (32768,128), (64,1),(32768,1)
        pos = torch.div(torch.mul(anchor, pos_pair), self.temp).sum(-1, keepdim=True) #(64,128)*(128,64)->(64,64) ->(64,1)

        mask_pixel_filter = (pseudo_label_1 != pseudo_label_all.squeeze(-1).unsqueeze(0)).float() #(64, 1)!=(1, 32768)->(64, 32768)
        mask_pixel_filter = torch.cat([torch.ones(mask_pixel_filter.size(0), 1).float().cuda(), mask_pixel_filter], 1) #(64, 1+32768)->(64, 32769)
        mask_patch_filter = (
                (fcs_1 + fcs_all.squeeze(-1).unsqueeze(0)) <= (self.bdp_threshold + self.fdp_threshold)).float() #(64, 1) + (1, 32768)->(64, 32768)
        mask_patch_filter = torch.cat([torch.ones(mask_patch_filter.size(0), 1).float().cuda(), mask_patch_filter], 1) #(64, 1+32768)->(64, 32769)

        neg = torch.div(torch.matmul(anchor, neg_pair.T), self.temp) #(64,128)*(128, 32768)->(64, 32768)
        neg = torch.cat([pos, neg], 1) #(64, 1+32768)->(64, 32769)
        max = torch.max(neg, dim=1, keepdim=True)[0] #(64, 1) #keepdim=True表示输出与输入维度相同，此例中都是二维
        exp_neg = (torch.exp(neg - max) * mask_pixel_filter * mask_patch_filter).sum(-1) #(64, 32769)->(64,),此时sum(-1)中默认keepdim=False即减少一个维度从二维变为一维

        loss = torch.exp(pos - max).squeeze(-1) / (exp_neg + self.eps) #(64,)/(64,) ->(64,)
        loss = -torch.log(loss + self.eps) #(64,)
        return loss

class ConsistencyWeight(nn.Module):
    def __init__(self, max_weight, max_epoch, ramp='sigmoid'):
        super().__init__()
        self.max_weight = max_weight
        self.max_epoch = max_epoch
        self.ramp = ramp

    def forward(self, epoch):
        current = np.clip(epoch, 0.0, self.max_epoch) #将epoch值限定在0~max_epoch之间
        phase = 1.0 - current / self.max_epoch
        if self.ramp == 'sigmoid':
            ramps = float(np.exp(-5.0 * phase * phase))
        elif self.ramp == 'log':
            ramps = float(1 - np.exp(-5.0 * current / self.max_epoch))
        elif self.ramp == 'exp':
            ramps = float(np.exp(5.0 * (current / self.max_epoch - 1)))
        else:
            ramps = 1.0

        consistency_weight = self.max_weight * ramps
        return consistency_weight

class ContrastiveLossSup(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ContrastiveLossSup, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.zeros((batch_size*width*width,1)).cuda()
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

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss