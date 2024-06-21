import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseContrastive(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.capacity = opts.capacity
        self.count = opts.count
        self.feature_bank = []
        self.label_bank = []
        self.FC_bank = []
        self.bdp_threshold = 0.2
        self.fdp_threshold = 0.6
        self.temp = 0.1
        self.contrastive_loss = ContrastiveLoss(self.bdp_threshold, self.fdp_threshold, self.temp)

    def forward(self, proj_main, proj_ema, label_main, label_ema, patch_num=8):
        b, c = proj_main.size(0), proj_main.size(1)
        h, w = proj_main.size(2) // patch_num, proj_main.size(3) // patch_num
        patches_features_main = []
        patches_features_ema = []
        patches_labels_main = []
        patches_labels_ema = []
        FC_main = []
        FC_ema = []

        for i in range(patch_num * patch_num):
            j = i // patch_num
            k = i % patch_num
            for ii in range(b):
                patch_main = proj_main[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]
                patch_ema = proj_ema[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]
                patch_label_main = label_main[ii, j * h: (j + 1) * h, k * w: (k + 1) * w]
                patch_label_ema = label_ema[ii, j * h: (j + 1) * h, k * w: (k + 1) * w]
                patches_features_main.append(patch_main)
                patches_features_ema.append(patch_ema)
                patches_labels_main.append(patch_label_main)
                patches_labels_ema.append(patch_label_ema)
                fc_main = patch_label_main.sum().item() / (h * w)
                fc_ema = patch_label_ema.sum().item() / (h * w)
                FC_main.append([fc_main] * h * w)
                FC_ema.append([fc_ema] * h * w)

        _patches_main = [p.permute(1, 2, 0).contiguous().view(h * w, c) for p in patches_features_main]
        _patches_ema = [p.permute(1, 2, 0).contiguous().view(h * w, c) for p in patches_features_ema]
        _patches_main = torch.cat(_patches_main, 0)
        _patches_ema = torch.cat(_patches_ema, 0)
        _patches_labels_main = [p.contiguous().view(-1) for p in patches_labels_main]
        _patches_labels_ema = [p.contiguous().view(-1) for p in patches_labels_ema]
        _patches_labels_main = torch.cat(_patches_labels_main, 0)
        _patches_labels_ema = torch.cat(_patches_labels_ema, 0)
        # _patches_features = torch.cat([_patches_main, _patches_ema], 0).detach()
        # _patches_labels = torch.cat([_patches_labels_main, _patches_labels_ema], 0).detach()
        _FC = torch.tensor(FC_main).view(-1).cuda()

        # features_all = _patches_main.detach()
        # labels_all = _patches_labels_main.detach()
        # FC_all = _FC
        #
        # self.feature_bank.append(features_all)
        # self.label_bank.append(labels_all)
        # self.FC_bank.append(FC_all)
        # if self.count > self.capacity:
        #     self.feature_bank = self.feature_bank[1:]#先进先出队列
        #     self.label_bank = self.label_bank[1:]
        #     self.FC_bank = self.FC_bank[1:]
        # else:
        #     self.count += 1
        # feature_all = torch.cat(self.feature_bank, 0).detach()
        # label_all = torch.cat(self.label_bank, 0).detach()
        # FC_all = torch.cat(self.FC_bank, 0).detach()

        loss_contr_sum = 0.0
        loss_contr_count = 0

        for i_patch in range(b * patch_num * patch_num):
            patch_main_i = patches_features_main[i_patch]
            patch_ema_i = patches_features_ema[i_patch]
            label_main_i = patches_labels_main[i_patch]
            label_ema_i = patches_labels_ema[i_patch]

            patch_main_i = patch_main_i.permute(1, 2, 0).contiguous().view(-1, c)
            patch_ema_i = patch_ema_i.permute(1, 2, 0).contiguous().view(-1, c)
            label_main_i = label_main_i.contiguous().view(-1)
            label_ema_i = label_ema_i.contiguous().view(-1)

            # fc = label_main_i.sum().item() / label_main_i.size(0)
            # if fc > self.bdp_threshold and fc < self.fdp_threshold:
            #     continue
            # else:
            #     loss_contr_count += 1
            #     FC_i = [fc] * h * w
            #     FC_i = torch.tensor(FC_i).cuda()

            loss_contr = torch.utils.checkpoint.checkpoint(self.contrastive_loss, patch_main_i, patch_ema_i,
                                                           _patches_ema, label_main_i, _patches_labels_ema)#checkpoint减少模型内存占用
            loss_contr = loss_contr.mean()
            loss_contr_sum += loss_contr
            loss_contr_count += 1

        loss_contr = loss_contr_sum / loss_contr_count
        return loss_contr

class ContrastiveLoss(nn.Module):
    def __init__(self, bdp_threshold, fdp_threshold, temp=0.1, eps=1e-8):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.eps = eps
        self.bdp_threshold = bdp_threshold
        self.fdp_threshold = fdp_threshold

    def forward(self, anchor, pos_pair, neg_pair, pseudo_label, pseudo_label_all):
        pos = torch.div(torch.mul(anchor, pos_pair), self.temp).sum(-1, keepdim=True)

        # mask_pixel_filter = (pseudo_label.unsqueeze(-1) != pseudo_label_all.unsqueeze(0)).float()
        # mask_patch_filter = ((FC.unsqueeze(-1) + FC_all.unsqueeze(0)) <= (self.bdp_threshold + self.fdp_threshold)).float()
        # mask_pixel_filter = torch.cat([torch.ones(mask_pixel_filter.size(0), 1).float().cuda(), mask_pixel_filter], 1)
        # mask_patch_filter = torch.cat([torch.ones(mask_patch_filter.size(0), 1).float().cuda(), mask_patch_filter], 1)

        neg = torch.div(torch.matmul(anchor, neg_pair.T), self.temp)
        neg = torch.cat([pos, neg], 1)
        max = torch.max(neg, 1, keepdim=True)[0]
        exp_neg = (torch.exp(neg - max)).sum(-1)

        loss = torch.exp(pos - max).squeeze(-1) / (exp_neg + self.eps)
        loss = -torch.log(loss + self.eps)

        return loss