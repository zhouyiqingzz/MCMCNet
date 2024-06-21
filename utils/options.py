import torch
import torch.nn as nn
import torch.nn.functional as F

class Opts(object):
    def __init__(self):
        self.mode= 'semi'
        self.ignore_index=255
        self.reduction= 'mean'
        self.epoch_semi=50
        self.in_dim=256
        self.out_dim=16
        self.downsample=True
        self.capacity=2
        self.count=0
        self.patch_num=8
        self.bdp_threshold=0.4
        self.fdp_threshold=0.6
        self.weight_contr=0.4
        self.weight_ent=0.2
        self.weight_cons=0.4
        self.max_epoch=200
        self.ramp='sigmoid'
        self.threshold=0.5
