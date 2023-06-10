import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.CSA_model import Conv

class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,interpolate=True):
        super(DecoderBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,in_channels//4,kernel_size=1)
        self.bn1=nn.BatchNorm2d(in_channels//4)
        self.relu1=nn.ReLU()
        self.interpolate=interpolate

        self.stConv1=nn.Conv2d(in_channels//4,in_channels//4,kernel_size=(1,9),padding=(0,4),groups=in_channels//4)
        self.stConv2=nn.Conv2d(in_channels//4,in_channels//4,kernel_size=(9,1),padding=(4,0),groups=in_channels//4)
        self.stConv3=nn.Conv2d(in_channels//4,in_channels//4,kernel_size=(9,1),padding=(4,0),groups=in_channels//4)
        self.stConv4=nn.Conv2d(in_channels//4,in_channels//4,kernel_size=(1,9),padding=(0,4),groups=in_channels//4)

        self.bn2=nn.BatchNorm2d(in_channels)
        self.relu2=nn.ReLU()
        self.conv3=nn.Conv2d(in_channels,out_channels,kernel_size=1)
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
        x=x.reshape(shape[0],shape[1],-1).contiguous()#另开一块内存的复制
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
        # x=self.bn1(x)
        x=self.relu1(x)

        x1=self.stConv1(x)#64*64*64
        x2=self.stConv2(x)#64*64*64
        # print(x1.shape,x2.shape)
        x3=self.stConv3(self.h_transform(x))
        x3=self.inv_h_transform(x3)#64*64*64
        x4=self.stConv4(self.v_transform(x))
        x4=self.inv_v_transform(x4)#64*64*64
        # print(x3.shape,x4.shape)
        x=torch.cat((x1,x2,x3,x4),1)#256*64*64
        #此处为False
        if self.interpolate:
            x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
        # x=self.bn2(x)
        # x=self.relu2(x)
        x=self.conv3(x)#out_channels*128*128 #可为128*128*128
        x=self.bn3(x)
        x=self.relu3(x)

        return x
