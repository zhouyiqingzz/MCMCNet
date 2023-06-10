import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aspp import *
from models.deform_conv import *
from models.strip_conv import *
from models.se_sk import *
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

#CFP模块
class CFPModule(nn.Module):
    def __init__(self,in_channels,k_size=3,dk_size=3,r=1):
        super().__init__()
        self.BN_PReLU_1=BNPReLU(in_channels)
        self.BN_PReLU_2=BNPReLU(in_channels)

        self.conv1x1_1=Conv(in_channels,in_channels//4,kernel_size=k_size,stride=1,padding=1,BN_act=True)

        #第四个分支(膨胀率为r+1)
        self.dconv3x1_4_1=Conv(in_channels//4,in_channels//16,kernel_size=(dk_size,1),stride=1,
                              padding=(1*r+1,0),dilation=(r+1,1),groups=in_channels//16,BN_act=True)
        self.dconv1x3_4_1=Conv(in_channels//16,in_channels//16,kernel_size=(1,dk_size),stride=1,
                              padding=(0,1*r+1),dilation=(1,r+1),groups=in_channels//16,BN_act=True)
        self.dconv3x1_4_2=Conv(in_channels//16,in_channels//16,kernel_size=(dk_size,1),stride=1,
                               padding=(1*r+1,0),dilation=(r+1,1),groups=in_channels//16,BN_act=True)
        self.dconv1x3_4_2=Conv(in_channels//16,in_channels//16,kernel_size=(1,dk_size),stride=1,
                               padding=(0,1*r+1),dilation=(1,r+1),groups=in_channels//16,BN_act=True)
        self.dconv3x1_4_3=Conv(in_channels//16,in_channels//8,kernel_size=(dk_size,1),stride=1,
                               padding=(1*r+1,0),dilation=(r+1,1),groups=in_channels//16,BN_act=True)
        self.dconv1x3_4_3=Conv(in_channels//8,in_channels//8,kernel_size=(1,dk_size),stride=1,
                               padding=(0,1*r+1),dilation=(1,r+1),groups=in_channels//8,BN_act=True)
        #第一个分支(膨胀率为1)
        self.dconv3x1_1_1 = Conv(in_channels // 4, in_channels // 16, kernel_size=(dk_size, 1), stride=1,
                                 padding=(1,0), groups=in_channels // 16, BN_act=True)
        self.dconv1x3_1_1 = Conv(in_channels // 16, in_channels // 16, kernel_size=(1, dk_size), stride=1,
                                 padding=(0,1), groups=in_channels // 16, BN_act=True)
        self.dconv3x1_1_2 = Conv(in_channels // 16, in_channels // 16, kernel_size=(dk_size, 1), stride=1,
                                 padding=(1,0), groups=in_channels // 16, BN_act=True)
        self.dconv1x3_1_2 = Conv(in_channels // 16, in_channels // 16, kernel_size=(1, dk_size), stride=1,
                                 padding=(0,1), groups=in_channels // 16, BN_act=True)
        self.dconv3x1_1_3 = Conv(in_channels // 16, in_channels // 8, kernel_size=(dk_size, 1), stride=1,
                                 padding=(1,0), groups=in_channels // 16, BN_act=True)
        self.dconv1x3_1_3 = Conv(in_channels // 8, in_channels // 8, kernel_size=(1, dk_size), stride=1,
                                 padding=(0,1), groups=in_channels // 8, BN_act=True)
        #第二个分支(膨胀率为r/4+1)
        self.dconv3x1_2_1 = Conv(in_channels // 4, in_channels // 16, kernel_size=(dk_size, 1), stride=1,
                                 padding=(int(r/4 + 1), 0), dilation=(int(r/4 + 1), 1), groups=in_channels // 16, BN_act=True)
        self.dconv1x3_2_1 = Conv(in_channels // 16, in_channels // 16, kernel_size=(1, dk_size), stride=1,
                                 padding=(0, int(r/4 + 1)), dilation=(1, int(r/4 + 1)), groups=in_channels // 16, BN_act=True)
        self.dconv3x1_2_2 = Conv(in_channels // 16, in_channels // 16, kernel_size=(dk_size, 1), stride=1,
                                 padding=(int(r/4 + 1), 0), dilation=(int(r/4 + 1), 1), groups=in_channels // 16, BN_act=True)
        self.dconv1x3_2_2 = Conv(in_channels // 16, in_channels // 16, kernel_size=(1, dk_size), stride=1,
                                 padding=(0, int(r/4 + 1)), dilation=(1, int(r/4 + 1)), groups=in_channels // 16, BN_act=True)
        self.dconv3x1_2_3 = Conv(in_channels // 16, in_channels // 8, kernel_size=(dk_size, 1), stride=1,
                                 padding=(int(r/4 + 1), 0), dilation=(int(r/4 + 1), 1), groups=in_channels // 16, BN_act=True)
        self.dconv1x3_2_3 = Conv(in_channels // 8, in_channels // 8, kernel_size=(1, dk_size), stride=1,
                                 padding=(0, int(r/4 + 1)), dilation=(1, int(r/4 + 1)), groups=in_channels // 8, BN_act=True)
        #第三个分支(膨胀率为r/2+1)
        self.dconv3x1_3_1 = Conv(in_channels // 4, in_channels // 16, kernel_size=(dk_size, 1), stride=1,
                                 padding=(int(r / 2 + 1), 0), dilation=(int(r / 2 + 1), 1), groups=in_channels // 16, BN_act=True)
        self.dconv1x3_3_1 = Conv(in_channels // 16, in_channels // 16, kernel_size=(1, dk_size), stride=1,
                                 padding=(0, int(r / 2 + 1)), dilation=(1, int(r / 2 + 1)), groups=in_channels // 16, BN_act=True)
        self.dconv3x1_3_2 = Conv(in_channels // 16, in_channels // 16, kernel_size=(dk_size, 1), stride=1,
                                 padding=(int(r / 2 + 1), 0), dilation=(int(r / 2 + 1), 1), groups=in_channels // 16, BN_act=True)
        self.dconv1x3_3_2 = Conv(in_channels // 16, in_channels // 16, kernel_size=(1, dk_size), stride=1,
                                 padding=(0, int(r / 2 + 1)), dilation=(1, int(r / 2 + 1)), groups=in_channels // 16, BN_act=True)
        self.dconv3x1_3_3 = Conv(in_channels // 16, in_channels // 8, kernel_size=(dk_size, 1), stride=1,
                                 padding=(int(r / 2 + 1), 0), dilation=(int(r / 2 + 1), 1), groups=in_channels // 16, BN_act=True)
        self.dconv1x3_3_3 = Conv(in_channels // 8, in_channels // 8, kernel_size=(1, dk_size), stride=1,
                                 padding=(0, int(r / 2 + 1)), dilation=(1, int(r / 2 + 1)), groups=in_channels // 8, BN_act=True)

        self.conv1x1_2=Conv(in_channels,in_channels,kernel_size=1,stride=1,padding=0,BN_act=False)

    def forward(self,input):
        input_1=self.BN_PReLU_1(input)
        input_1=self.conv1x1_1(input_1)

        output_1_1=self.dconv3x1_1_1(input_1)
        output_1_1=self.dconv1x3_1_1(output_1_1)
        output_1_2=self.dconv3x1_1_2(output_1_1)
        output_1_2=self.dconv1x3_1_2(output_1_2)
        output_1_3=self.dconv3x1_1_3(output_1_2)
        output_1_3=self.dconv1x3_1_3(output_1_3)

        output_2_1 = self.dconv3x1_2_1(input_1)
        output_2_1 = self.dconv1x3_2_1(output_2_1)
        output_2_2 = self.dconv3x1_2_2(output_2_1)
        output_2_2 = self.dconv1x3_2_2(output_2_2)
        output_2_3 = self.dconv3x1_2_3(output_2_2)
        output_2_3 = self.dconv1x3_2_3(output_2_3)

        output_3_1 = self.dconv3x1_3_1(input_1)
        output_3_1 = self.dconv1x3_3_1(output_3_1)
        output_3_2 = self.dconv3x1_3_2(output_3_1)
        output_3_2 = self.dconv1x3_3_2(output_3_2)
        output_3_3 = self.dconv3x1_3_3(output_3_2)
        output_3_3 = self.dconv1x3_3_3(output_3_3)

        output_4_1 = self.dconv3x1_4_1(input_1)
        output_4_1 = self.dconv1x3_4_1(output_4_1)
        output_4_2 = self.dconv3x1_4_2(output_4_1)
        output_4_2 = self.dconv1x3_4_2(output_4_2)
        output_4_3 = self.dconv3x1_4_3(output_4_2)
        output_4_3 = self.dconv1x3_4_3(output_4_3)

        output_1=torch.cat([output_1_1,output_1_2,output_1_3],1)
        output_2=torch.cat([output_2_1,output_2_2,output_2_3],1)
        output_3=torch.cat([output_3_1,output_3_2,output_3_3],1)
        output_4=torch.cat([output_4_1,output_4_2,output_4_3],1)

        output_add_1=output_1
        output_add_2=output_add_1+output_2
        output_add_3=output_add_2+output_3
        output_add_4=output_add_3+output_4
        output=torch.cat([output_add_1,output_add_2,output_add_3,output_add_4],1)
        output=self.BN_PReLU_2(output)
        output=self.conv1x1_2(output)

        return output+input

#下采样模块
class DownSampleBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        if self.in_channels<self.out_channels:
            self.n_channels=self.out_channels-self.in_channels
        else:
            self.n_channels=self.out_channels
        self.conv3x3=Conv(self.in_channels,self.n_channels,kernel_size=3,stride=2,padding=1)
        self.max_pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.BN_PReLU=BNPReLU(out_channels)

    def forward(self,input):
        output=self.conv3x3(input)
        max_pool = self.max_pool(input)
        if self.in_channels<self.out_channels:
            output=torch.cat([output,max_pool],1)
        output=self.BN_PReLU(output)

        return output

#三次原始输入图片平均池化
class InputInjection(nn.Module):
    def __init__(self,times):
        super().__init__()
        self.pool=nn.ModuleList()
        for i in range(times):
            self.pool.append(nn.AvgPool2d(3,stride=2,padding=1))#每一次平均池化尺寸降为原来的1/2

    def forward(self,input):
        for pool in self.pool:
            input=pool(input)
        return input

#CSA注意力模块
class CSAModule(nn.Module):
    def __init__(self,image_h_channel,image_l_channel):
        super().__init__()

        self.image_h_channel=image_h_channel
        self.image_l_channel=image_l_channel

        self.conv3x3_V1=Conv(self.image_h_channel,self.image_l_channel,kernel_size=3,stride=1,padding=1)
        self.conv3x3_K1=Conv(self.image_h_channel,self.image_l_channel,kernel_size=3,stride=1,padding=1)
        self.conv3x3_Q1=Conv(self.image_l_channel,self.image_l_channel,kernel_size=3,stride=1,padding=1)
        self.conv3x3_Q2=Conv(self.image_l_channel,self.image_l_channel,kernel_size=3,stride=1,padding=1)
        self.deconv3x3_Q1=DeConv(self.image_l_channel,self.image_l_channel,kernel_size=3,stride=1,padding=1,output_padding=0)
        self.conv3x3_V2=Conv(self.image_l_channel,self.image_l_channel,kernel_size=3,stride=1,padding=1)
        self.conv3x3_K2=Conv(self.image_l_channel,self.image_l_channel,kernel_size=3,stride=1,padding=1)

    def forward(self,input_h,input_l):
        V1=self.conv3x3_V1(input_h)#C,H,W
        K1=self.conv3x3_K1(input_h)
        Q1=self.conv3x3_Q1(input_l)
        Q2=self.conv3x3_Q2(input_l)#C,H,W
        Q1_u=F.interpolate(Q1,(Q1.size()[2],K1.size()[3]),mode='bilinear',align_corners=False)#N,C,H,W
        #print(Q1_u.shape,K1.shape)
        K1_mulpily_Q1=torch.matmul(Q1_u.permute(0,3,2,1),K1.permute(0,3,1,2))#W,H,H
        #print(K1_mulpily_Q1.shape)#W,H,H
        S1=F.softmax(K1_mulpily_Q1,3)#Wh,Hl,Hh
        M1=torch.matmul(S1,V1.permute(0,3,2,1)).permute(0,2,1,3)#Wh,Hl,Hh * Wh,Hh,Cl ->Wh,Hl,Cl -> Hl,Wh,Cl
        #print(M1.shape)

        V2=self.conv3x3_V2(M1.permute(0,3,1,2))#C,H,W
        K2=self.conv3x3_K2(M1.permute(0,3,1,2))#C,H,W
        #print(K2.shape,V2.shape)
        K2_multiply_Q2=torch.matmul(Q2.permute(0,2,3,1),K2.permute(0,2,1,3))#Hl,Wl,Cl * Hl,Cl,Wh -> Hl,Wl,Wh
        #print(K2_multiply_Q2.shape)
        S2=F.softmax(K2_multiply_Q2,3)# Hl,Wl,Wh
        #print(S2.shape)
        M2=torch.matmul(S2,V2.permute(0,2,3,1)).permute(0,3,1,2)#Hl,Wl,Cl->Cl,Hl,Wl
        #print(M2.shape)
        output=input_l+M2
        #print(output.shape)
        return output


#整个CSANet网络结构
class CSANet(nn.Module):
    def __init__(self,n_classes=2,Block_1_num=2,Block_2_num=6,Block_3_num=6):
        super().__init__()
        self.Init_conv=nn.Sequential(
            Conv(3,32,kernel_size=3,stride=2,padding=1,BN_act=True),
            Conv(32,32,kernel_size=3,stride=1,padding=1,BN_act=True),
            Conv(32,32,kernel_size=3,stride=1,padding=1,BN_act=True),
        )
        self.origin_down_1=InputInjection(1)
        self.origin_down_2=InputInjection(2)
        self.origin_down_3=InputInjection(3)
        self.origin_down_4 = InputInjection(4)

        #第1个CFP集群
        self.BN_PReLU_1=BNPReLU(32+3)
        self.Block_1_dilationRatio=[2,2]
        self.downSample_1=DownSampleBlock(32+3,64)#下采样前的通道数为初始卷积与原始图片平均池化后之和
        self.CFP_Block_1=nn.Sequential()
        for i in range(Block_1_num):
            self.CFP_Block_1.add_module("CFP_Module_1"+str(i),CFPModule(64,r=self.Block_1_dilationRatio[i]))
        self.BN_PReLU_2=BNPReLU(128+3)
        self.CSAModule_1 = CSAModule(32, 128+3)
        self.conv1x1_1=Conv(128+3,128,kernel_size=1,stride=1,padding=0,BN_act=True)

        #第2个CFP集群
        self.Block_2_dilationRatio = [4,4,8,8,16,16]
        self.downSample_2 = DownSampleBlock(128, 128)  # 下采样前的通道数为初始卷积与原始图片平均池化后之和
        self.CFP_Block_2 = nn.Sequential()
        for i in range(Block_2_num):
            self.CFP_Block_2.add_module("CFP_Module_2"+str(i),CFPModule(128,r=self.Block_2_dilationRatio[i]))
        self.BN_PReLU_3 = BNPReLU(256 + 3)
        self.CSAModule_2 = CSAModule(128,256+3)
        self.conv1x1_2=Conv(256+3,256,kernel_size=1,stride=1,padding=0,BN_act=True)#有待修改

        #第3个CFP集群
        self.Block_3_dilationRatio = [4, 4, 8, 8, 16, 16]
        self.downSample_3 = DownSampleBlock(256, 256)  # 下采样前的通道数为初始卷积与原始图片平均池化后之和
        self.CFP_Block_3 = nn.Sequential()
        for i in range(Block_3_num):
            self.CFP_Block_3.add_module("CFP_Module_3" + str(i), CFPModule(256, r=self.Block_3_dilationRatio[i]))
        self.BN_PReLU_4 = BNPReLU(512 + 3)
        self.CSAModule_3 = CSAModule(256 , 512 + 3)
        self.conv1x1_3 = Conv(512 + 3, 512, kernel_size=1, stride=1, padding=0, BN_act=True)  # 有待修改

        self.ASPP=ASPP(512,512)
        self.SELayer=SENet(512)
        self.prelu=nn.PReLU()
        # self.deconv2x2_3=DeConv(128 + 128, 64, kernel_size=3, stride=2, padding=1, output_padding=1,BN_act=True) # 64*128*128,output = (input-1)stride+outputpadding -2padding+kernelsize
        self.conv1x1_d3=Conv(512+3,256,kernel_size=1,stride=1,padding=0,BN_act=True)
        # self.deconv2x2_2=DeConv(64 + 64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, BN_act=True)# 32*256*256
        self.conv1x1_d2=Conv(256+3,128,kernel_size=1,stride=1,padding=0,BN_act=True)
        # self.deconv2x2_1=DeConv(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, BN_act=True)# 16*512*512
        self.conv1x1_d1=Conv(64,16,kernel_size=1,stride=1,padding=0,BN_act=True)
        self.conv1x1_d0 = Conv(32, 16, kernel_size=1, stride=1, padding=0, BN_act=True)

        self.decoder_3=DecoderBlock(512,256)
        self.decoder_2=DecoderBlock(256,128)
        self.decoder_1 = DecoderBlock(128, 64)
        self.decoder_0 = DecoderBlock(64, 32)
        self.deconv2x2_1 = DeConv(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1,
                                  BN_act=True)  # 16*512*512
        self.conv1x1_last=Conv(16, 1, kernel_size=1, stride=1, padding=0, BN_act=False)# 1*512*512
        self.sigmoid=nn.Sigmoid()

    def forward(self,input):#输入:(3,512,512)
        #print(input.size())
        encode_0=self.Init_conv(input)#(32,256,256)#作为e1应用于Decoder第一阶段
        # print(output_0.size())
        origin_down_1 = self.origin_down_1(input)#(3,256,256)#与上面encode_0形成分支
        origin_down_2 = self.origin_down_2(input)#(3,128,128)
        origin_down_3 = self.origin_down_3(input)#(3,64,64)
        origin_down_4 = self.origin_down_4(input)#(3,32,32)
        encode_0_cat=self.BN_PReLU_1(torch.cat([encode_0,origin_down_1],1))#(64+3,256,256)

        #第1次下采样及CFP集群后的输出
        encode_1_1=self.downSample_1(encode_0_cat)#(64,128,128) #作为e2应用于Decoder第二阶段
        encode_1_2=self.CFP_Block_1(encode_1_1)#(64,128,128)
        encode_1_cat=self.BN_PReLU_2(torch.cat([encode_1_2,encode_1_1,origin_down_2],1))#(128+3,128,128)
        encode_1_attn=self.CSAModule_1(encode_0,encode_1_cat)#(128+3,128,128)
        encode_1_sum=encode_1_cat+encode_1_attn #(128+3,128,128)
        encode_1=self.conv1x1_1(encode_1_sum)#(128,128,128)#此处有待多观察!!!!可能是自己随便加的
        # print(encode_1.shape)

        #第2次下采样及CFP集群后的输出
        encode_2_1 = self.downSample_2(encode_1)#(128,64,64) #作为e3，应用于Decoder第三阶段
        # print(encode_2_1.shape)
        encode_2_2 = self.CFP_Block_2(encode_2_1)#(128,64,64)
        # print(encode_2_2.shape,origin_down_3.shape)
        encode_2_cat = self.BN_PReLU_3(torch.cat([encode_2_2, encode_2_1, origin_down_3], 1))#(256+3,64,64)
        encode_2_attn=self.CSAModule_2(encode_1,encode_2_cat)#(256+3,64,64)
        encode_2_sum=encode_2_cat+encode_2_attn#(256+3,64,64)
        encode_2=self.conv1x1_2(encode_2_sum)#(256,64,64) #作为e4，应用于Decoder第四阶段
        # print(encode_2.shape)

        # 第3次下采样及CFP集群后的输出
        encode_3_1 = self.downSample_3(encode_2)  # (256,32,32) #作为e3，应用于Decoder第三阶段
        encode_3_2 = self.CFP_Block_3(encode_3_1)  # (256,32,32)
        encode_3_cat = self.BN_PReLU_4(torch.cat([encode_3_2, encode_3_1, origin_down_4], 1))  # (256+256+3,32,32)
        encode_3_attn = self.CSAModule_3(encode_2, encode_3_cat)  # (512+3,32,32)
        encode_3_sum = encode_3_cat + encode_3_attn  # (512+3,32,32)
        encode_3 = self.conv1x1_3(encode_3_sum)  # (512,32,32) #作为e4，应用于Decoder第四阶段
        # print(encode_3.shape)
        bottle_branch_1=self.ASPP(encode_3)#(512,32,32)
        bottle_branch_2=self.SELayer(encode_3)#(512,32,32)
        bottle=bottle_branch_1+bottle_branch_2
        # print(bottle.shape)
        # print(bottle.shape)
        #采用反卷积或双线性插值进行上采样
        decode_3=self.decoder_3(bottle)#(256,64,64)
        decode_3=decode_3+encode_2#(256,64,64)
        # print(decode_3.shape)
        # decode_3=self.conv1x1_d3(decode_3)#(256,64,64)

        decode_2=self.decoder_2(decode_3)#(128,128,128)
        decode_2=decode_2+encode_1#(128,128,128)
        # print(decode_2.shape)
        # decode_2=self.conv1x1_d2(decode_2)#(128,128,128)

        decode_1 = self.decoder_1(decode_2)  # (64,256,256)
        # print(decode_1.shape)
        # decode_1 = self.conv1x1_d1(decode_1)  # (16,256,256)

        # decode_0 = self.decoder_0(decode_1)  # (32,512,512)
        # decode_0 = self.conv1x1_d0(decode_0)  # (16,512,512)
        decode_0=self.deconv2x2_1(decode_1)#(16,512,512)
        # print(decode_0.shape)
        output=self.conv1x1_last(decode_0)#(1,512,512)
        output_final=self.sigmoid(output)
        return output_final

#注：可尝试使用深度可分离卷积，主要包含深度卷积(每个通道分别卷积)和逐点全通道卷积两个模块，几乎不影响精度的情况下参数量下降至原来1/9
if __name__=='__main__':
    model=CSANet()
    x=torch.randn((4,3,512,512))
    output=model(x)
    print(output.shape)
