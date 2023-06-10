import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPModule(nn.Module):#空洞空间金字塔池化
    def __init__(self,in_channels,out_channels,kernel_size,padding,dilation):
        super(ASPPModule,self).__init__()
        self.astrous_conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=1,
                                    padding=padding,dilation=dilation,bias=False)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        x=self.astrous_conv(x)
        # x=self.bn(x)
        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ASPP,self).__init__()
        dilations=[1,6,12,18]
        self.aspp1=ASPPModule(in_channels=in_channels,out_channels=256,kernel_size=1,padding=0,
                              dilation=dilations[0])
        self.aspp2=ASPPModule(in_channels=in_channels,out_channels=256,kernel_size=3,padding=dilations[1],
                              dilation=dilations[1])
        self.aspp3=ASPPModule(in_channels=in_channels, out_channels=256, kernel_size=3, padding=dilations[2],
                                dilation=dilations[2])
        self.aspp4=ASPPModule(in_channels=in_channels, out_channels=256, kernel_size=3, padding=dilations[3],
                                dilation=dilations[3])

        self.global_avg_pool=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),#(1,1)表示最终得到结果的尺寸
            nn.Conv2d(in_channels,256,kernel_size=1,stride=1,bias=False),
            # nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv1=nn.Conv2d(1280,out_channels,kernel_size=1,bias=False)#(256*5)->256
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.5)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):#input:256*64*64
        x1=self.aspp1(x)#256*64*64
        x2=self.aspp2(x)#256*64*64
        x3=self.aspp3(x)#256*64*64
        x4=self.aspp4(x)#256*64*64
        x5=self.global_avg_pool(x)#256*1*1
        x5=F.interpolate(x5,size=x4.size()[2:],mode='bilinear',align_corners=True)#256*64*64
        x=torch.cat((x1,x2,x3,x4,x5),dim=1)#1280*64*64

        x=self.conv1(x)#256*64*64
        x=self.bn1(x)
        x=self.relu(x)
        output=self.dropout(x)#256*64*64
        return output

# model=ASPP(512,512)
# x=torch.randn((2,512,32,32))
# output=model(x)
# print(output.shape)