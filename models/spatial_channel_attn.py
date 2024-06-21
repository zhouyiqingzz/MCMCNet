import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)#此处的cuda给删了，请注意恢复

class CC_module(nn.Module):
    def __init__(self, in_dim):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()  # nChw

        proj_query = self.query_conv(x)  # nchw
        # nchw->nwch->(n*w)ch->(n*w)hc
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width,-1,height).permute(0,2,1)
        # nchw->nhcw->(n*h)*c*w>(n*h)wc
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height,-1,width).permute(0,2,1)

        proj_key = self.key_conv(x)  # nchw
        # nchw->nwch->(n*w)ch
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # nchw->nhcw->(n*h)cw
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x)  # nChw
        # nChw->nwCh->(n*w)Ch
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # nChw->nhCw->(n*h)Cw
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        # (n*w)hh->nwhh->nhwh
        energy_H=(torch.bmm(proj_query_H,proj_key_H)+self.INF(m_batchsize,height,width)).contiguous().view(m_batchsize,width,height,height).permute(0,2,1,3)
        # (n*h)ww->nhww
        energy_W = torch.bmm(proj_query_W, proj_key_W).contiguous().view(m_batchsize, height, width, width)

        # nhwh->nwhh->(n*w)hh
        att_H = self.softmax(energy_H).permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # nhww->(n*h)ww
        att_W = self.softmax(energy_W).contiguous().view(m_batchsize * height, width, width)
        # (n*w)Ch->nwCh->nChw
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).contiguous().view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        # (n*h)Cw->nhCw->nChw
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).contiguous().view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # nchw->n,c,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # n,c,1,1->n,c,1->n,1,c->n,c,1,1
        y = self.sigmoid(y)

        return  x * y.expand_as(x)