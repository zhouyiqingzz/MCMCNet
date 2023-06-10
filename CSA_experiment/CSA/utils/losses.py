import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class dice_bce_loss(nn.Module):
    def __init__(self,batch=True):
        super(dice_bce_loss,self).__init__()
        self.batch=batch
        self.bce_loss=nn.BCELoss()
        self.focalLoss2d=FocalLoss2d()

    def soft_dice_coeff(self,y_true,y_pred):
        smooth=0.0001
        if self.batch:
            i=torch.sum(y_true)
            j=torch.sum(y_pred)
            intersection=torch.sum(y_true*y_pred)
        else:
            i=y_true.sum(1).sum(1).sum(1)
            j=y_pred.sum(1).sum(1).sum(1)
            intersection=(y_true*y_pred).sum(1).sum(1).sum(1)
        score=(2.0*intersection+smooth)/(i+j+smooth)
        return score.mean()

    def soft_dice_loss(self,y_true,y_pred):
        loss=1.0-self.soft_dice_coeff(y_true,y_pred)
        return loss

    def __call__(self,y_true,y_pred):

        a=self.bce_loss(y_pred,y_true)
        b=self.soft_dice_loss(y_true,y_pred)
        # c=self.focalLoss2d(y_pred,y_true)
        return a+b

class CrossEntropyLoss2d(nn.Module):
    def __init__(self,weight=torch.tensor([0.9,0.1]).cuda(),ignore_label=0):
        super().__init__()
        self.loss=nn.NLLLoss2d(weight=weight,ignore_index=ignore_label)

    def forward(self,outputs,targets):
        return self.loss(F.log_softmax(outputs),torch.argmax(targets,1))


class FocalLoss2d(nn.Module):
    def __init__(self,alpha=0.5,gamma=2,ignore_index=0):
        super().__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.ignore_index=ignore_index
        self.CE_FN=nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self,outputs,targets):
        outputs=torch.tensor(outputs,dtype=torch.float)
        targets=torch.tensor(targets,dtype=torch.float)
        log_pt=-self.CE_FN(outputs,targets.long().squeeze(1))
        pt=torch.exp(log_pt)
        loss=-((1-pt)**self.gamma)*self.alpha*log_pt
        print(loss)
        return loss

class OhemCrossEntropy2dTensor(nn.Module):
    def __init__(self, ignore_label, reduction='elementwise_mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=True,weight=None):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        self.weight=weight
        if use_weight:
            # weight = torch.FloatTensor(
            #     [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
            #      0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
            #      1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

def cal_entropy_img(gx):
    smooth = 1e-8
    ex = -1.0 * gx*torch.log(gx + smooth) - 1.0 * (1-gx)*torch.log(1-gx+smooth)
    return ex





