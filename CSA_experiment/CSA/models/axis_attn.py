import torch
import torch.nn as nn
import torch.nn.functional as F


class RowAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RowAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()

        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        row_attn = torch.bmm(Q, K)
        row_attn = self.softmax(row_attn)

        out = torch.bmm(V, row_attn.permute(0, 2, 1))
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)
        out = self.gamma * out + x

        return out

class ColAttention(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ColAttention,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.query_conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1)
        self.key_conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1)
        self.value_conv=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1)
        self.softmax=nn.Softmax(dim=2)
        self.gamma=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        b,_,h,w=x.size()

        Q=self.query_conv(x)
        K=self.key_conv(x)
        V=self.value_conv(x)

        Q=Q.permute(0,3,1,2).contiguous().view(b*w,-1,h).permute(0,2,1)
        K=K.permute(0,3,1,2).contiguous().view(b*w,-1,h)
        V=V.permute(0,3,1,2).contiguous().view(b*w,-1,h)

        col_attn=torch.bmm(Q,K)
        col_attn=self.softmax(col_attn)

        out=torch.bmm(V,col_attn.permute(0,2,1))
        out=out.view(b,w,-1,h).permute(0,2,3,1)
        out=self.gamma*out+x

        return out

x=torch.randn(16,3,256,256)
r_attn=RowAttention(3,2)
c_attn=ColAttention(3,2)
x=r_attn(x)
y=c_attn(x)
print(y.shape)