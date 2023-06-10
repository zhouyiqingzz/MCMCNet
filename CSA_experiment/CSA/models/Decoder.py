import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from modeling.sync_batchnorm.batchnorm import SynchronizeBatchNorm2d

class SENet(nn.Module):
    def __init__(self, in_channels, reduction=3):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,inp=False):
        super(DecoderBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,in_channels//4,kernel_size=1)
        self.bn1=nn.BatchNorm2d(in_channels//4)
        self.relu1=nn.ReLU()
        self.inp=inp

        self.stConv1=nn.Conv2d(in_channels//4,in_channels//8,kernel_size=(1,9),padding=(0,4))
        self.stConv2=nn.Conv2d(in_channels//4,in_channels//8,kernel_size=(9,1),padding=(4,0))
        self.stConv1=nn.Conv2d(in_channels//4,in_channels//8,kernel_size=(9,1),padding=(4,0))
        self.stConv1=nn.Conv2d(in_channels//4,in_channels//8,kernel_size=(1,9),padding=(0,4))

        self.bn2=nn.BatchNorm2d((in_channels//8)*4)
        self.relu2=nn.ReLU()
        self.conv3=nn.Conv2d((in_channels//8)*4,out_channels,kernel_size=1)
        self.bn3=nn.BatchNorm2d(out_channels)
        self.relu3=nn.ReLU()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self,x):#与inv_h_transform(self,x)函数共同构成左下右上条状卷积
        shape=x.size()
        x=torch.nn.functional.pad(x,(0,shape[-1]))
        x=x.reshape(shape[0],shape[1],-1)[..., :-shape[-1]]
        x=x.reshape(shape[0],shape[1],shape[2],2*shape[3]-1)

        return x

    def inv_h_transform(self,x):
        shape=x.size()
        x=x.reshape(shape[0],shape[1],-1).continuous()#另开一块内存的复制
        x=torch.nn.functional.pad(x,(0,shape[-2]))
        x=x.reshape(shape[0],shape[1],shape[-2],2*shape[-2])
        x=x[..., 0:shape[-2]]

        return x

    def v_transform(self,x):#与inv_v_transform(self,x)共同构成坐上右下条状卷积
        x=x.permute(0,1,3,2)
        shape=x.size()
        x=torch.nn.functional.pad(x,(0,shape[-1]))
        x=x.reshape(shape[0],shape[1],-1)[..., :-shape[-1]]
        x=x.reshape(shape[0],shape[1],shape[2],2*shape[3]-1)

        return x.permute(0,1,3,2)

    def inv_v_transform(self,x):
        x=x.permute(0,1,3,2)
        shape=x.size()
        x=x.reshape(shape[0],shape[1],-1)
        x=torch.nn.functional.pad(x,(0,shape[-2]))
        x=x.reshape(shape[0],shape[1],shape[-2],2*shape[-2])
        x=x[..., 0:shape[-2]]

        return x.permute(0,1,3,2)

    def forward(self,x):#输入:256*64*64
        x=self.conv1(x)#64*64*64
        x=self.bn1(x)
        x=self.relu1(x)

        x1=self.stConv1(x)#32*64*64
        x2=self.stConv2(x)#32*64*64
        x3=self.inv_h_transform(self.stConv3(self.h_transform(x)))#32*64*64
        x4=self.inv_v_transform(self.stConv4(self.h_transform(x)))#32*64*64
        x=torch.cat((x1,x2,x3,x4),1)#128*64*64
        if self.inp:
            x=F.interpolate(x,scale_factor=2,mode='bilinear')
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.conv3(x)#out_channels*32*32 #可为128*64*64
        x=self.bn3(x)
        x=self.relu3(x)

        return x

class Decoder(nn.Module):
    def __init__(self,in_channels=256,out_channels=256):
        super(Decoder,self).__init__()
        self.decoder4=DecoderBlock(in_channels,256)
        self.decoder3=DecoderBlock(512,128)
        self.decoder2=DecoderBlock(256,64,inp=True)
        self.decoder1=DecoderBlock(128,64,inp=True)

        self.conv_e3=nn.Sequential(
            nn.Conv2d(1024,256,kernel_size=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_e2=nn.Sequential(
            nn.Conv2d(512,128,kernel_size=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv_e1=nn.Sequential(
            nn.Conv2d(256,64,kernel_size=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,input,e1,e2,e3,e4):#e1:32*256*256,e2:64*128*128,e3:128*64*64,e4:256*32*32
        d4=torch.cat((self.decoder4(e4),self.conv_e3(e3)),dim=1)#64*64*64 cat 256*64*64
        d3=torch.cat((self.decoder3(d4),self.conv_e2(e2)),dim=1)#64*64*64 cat 128*64*64
        d2=torch.cat((self.decoder2(d3),self.conv_e1(e1)),dim=1)#128*128*128 cat 64*128*128
        d1=self.decoder1(d2)#128*256*256
        output=F.interpolate(d1,input.size()[2:],mode='bilinear',align_corner=True)#128*512*512

        return output
