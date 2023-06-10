import torch
import torch.nn as nn
import torch.nn.functional as F

def INF(B,H,W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    def __init__(self,in_channels):
        super(CrissCrossAttention,self).__init__()
        self.query_conv=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//8,kernel_size=1)
        self.key_conv=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//8,kernel_size=1)
        self.value_conv=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1)
        self.softmax=nn.Softmax(dim=3)
        self.INF=INF
        self.gamma=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        b,_,h,w=x.size()

        proj_query=self.query_conv(x)
        proj_query_H=proj_query.permute(0,3,1,2).contiguous().view(b*w,-1,h).permute(0,2,1)
        proj_query_W=proj_query.permute(0,2,1,3).contiguous().view(b*h,-1,w).permute(0,2,1)

        proj_key=self.key_conv(x)
        proj_key_H=proj_key.permute(0,3,1,2).contiguous().view(b*w,-1,h)
        proj_key_W=proj_key.permute(0,2,1,3).contiguous().view(b*w,-1,w)

        proj_value=self.value_conv(x)
        proj_value_H=proj_value.permute(0,3,1,2).contiguous().view(b*w,-1,h)
        proj_value_W=proj_value.permute(0,2,1,3).contiguous().view(b*h,-1,w)

        energy_H=(torch.bmm(proj_query_H,proj_key_H)+self.INF(b,h,w)).view(b,w,h,h).permute(0,2,1,3)
        energy_W=torch.bmm(proj_query_W,proj_key_W).view(b,h,w,w)
        concate=self.softmax(torch.cat([energy_H,energy_W],3))

        attn_H=concate[:,:,:,0:h].permute(0,2,1,3).contiguous().view(b*w,h,h)
        attn_W=concate[:,:,:,h:h+w].contiguous().view(b*h,w,w)
        out_H=torch.bmm(proj_value_H,attn_H.permute(0,2,1)).view(b,w,-1,h).permute(0,2,3,1)
        out_W=torch.bmm(proj_value_W,attn_W.permute(0,2,1)).view(b,h,-1,w).permute(0,2,1,3)

        return self.gamma*(out_H+out_W)+x

model=CrissCrossAttention(16)
x=torch.randn((16,16,128,128))
out=model(x)
print(x.shape)