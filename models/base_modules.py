import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu)#若inplace=True则变量在前向计算时是一个值，在求梯度时变成另一个值
#自定义的Conv类
class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation=(1,1),groups=1,BN_act=False,bias=False):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,
                            padding=padding,dilation=dilation,groups=groups,bias=bias)
        self.BN_act = BN_act
        if self.BN_act:
            self.BN_PReLu=BNPReLU(out_channels)

    def forward(self,input):
        output=self.conv(input)
        if self.BN_act:
            output=self.BN_PReLu(output)
        return output

#深度可分离卷积
class DepthWiseConv(nn.Module):
    def __init__(self,in_channels,out_channels,BN_act=True):
        super(DepthWiseConv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,
                                    stride=1,padding=1,groups=in_channels)#每个滤波器一个卷积核负责一个通道得到一个通道特征图
        self.point_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,
                                    stride=1,padding=0,groups=1)##每个滤波器多个卷积核(1x1)负责所有通道得到一个通道特征图，共得到out_channels个通道特征图
        self.BN_act = BN_act
        if self.BN_act:
            self.BN_PReLu=BNPReLU(out_channels)
    def forward(self,input):
        output = self.depth_conv(input)
        output = self.point_conv(output)
        if self.BN_act:
            output=self.BN_PReLu(output)
        return output

#自定义的DeConv类
class DeConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,output_padding,dilation=(1,1),groups=1,BN_act=False,bias=False):
        super().__init__()
        self.convT=nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,
                                      output_padding=output_padding,dilation=dilation,groups=groups,bias=bias)
        self.BN_act = BN_act
        if self.BN_act:
            self.BN_PReLU=BNPReLU(out_channels)

    def forward(self,input):
        output=self.convT(input)
        if self.BN_act:
            output=self.BN_PReLU(output)
        return output

#批处理+激活
class BNPReLU(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.BN=nn.BatchNorm2d(in_channels,1e-3)
        self.act=nn.PReLU(in_channels)
    def forward(self,input):
        output=self.BN(input)
        output=self.act(output)
        return output