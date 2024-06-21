import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial
from models.strip_conv import StripConvBlock
from models.spatial_channel_attn import CC_module,eca_layer
from models.Sea_Attn import Sea_Attention
torch.manual_seed(1)
nonlinearity = partial(F.relu, inplace=False)

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DLinkNet(nn.Module):
    def __init__(self, num_classes=2, pool_old=1):
        super().__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)#此处命令窗口会下载官方的resnet模型的.pth文件
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        if pool_old:
            self.firstmaxpool = resnet.maxpool
        else:
            self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=True)

        self.encoder1 = resnet.layer1
        self.eca1=eca_layer(64)
        self.cc1=CC_module(64)

        self.encoder2 = resnet.layer2
        self.eca2 = eca_layer(128)
        self.cc2 = CC_module(128)

        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.strip_conv1 = StripConvBlock(64, 64)
        self.strip_conv2 = StripConvBlock(128, 128)
        self.strip_conv3 = StripConvBlock(256, 256)
        # self.sea_attn1 = Sea_Attention(64, 64)
        # self.sea_attn3 = Sea_Attention(256, 256)

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(64, filters[0])

        self.projector = Projectors(in_channels=64, out_channels=32)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3 + pool_old, 2, pool_old)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 4 - pool_old, padding=pool_old)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=pool_old)

        self.ske_decoder2 = DecoderBlock(filters[1], filters[0])
        self.ske_decoder1 = DecoderBlock(64, filters[0])
        self.ske_finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3 + pool_old, 2, pool_old)
        self.ske_finalrelu1 = nonlinearity
        self.ske_finalconv2 = nn.Conv2d(32, 32, 4 - pool_old, padding=pool_old)
        self.ske_finalrelu2 = nonlinearity
        self.ske_finalconv3 = nn.Conv2d(32, num_classes, 3, padding=pool_old)

        # 与上面的ske解码器二选一
        # self.ske_conv1 = nn.Conv2d(448, 64, kernel_size=3, stride=1, padding=1) #注意输入通道数的修改
        # self.ske_finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3 + pool_old, 2, pool_old)
        # self.ske_finalrelu1 = nonlinearity
        # self.ske_finalconv2 = nn.Conv2d(32, 32, 4 - pool_old, padding=pool_old)
        # self.ske_finalrelu2 = nonlinearity
        # self.ske_finalconv3 = nn.Conv2d(32, num_classes, 3, padding=pool_old)

    def forward(self, x):  # 4*3*512*512
        # Encoder
        x = self.firstconv(x)  # 4*64*256*256
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # 4*64*128*128

        e1 = self.encoder1(x)  # 4*64*128*128
        # e1 = self.strip_conv1(e1) # 4*64*128*128
        # e1 = self.eca1(e1)
        # e1 = self.cc1(e1)

        e2 = self.encoder2(e1)  # 4*128*64*64
        # e2 = self.strip_conv2(e2) # 4*128*64*64

        e3 = self.encoder3(e2)  # 4*256*32*32
        l3 = self.strip_conv3(e3) # 4*256*32*32

        e4 = self.encoder4(e3)  # 4*512*16*16

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + l3  # 4*256*32*32
        d3 = self.decoder3(d4) + e2  # 4*128*64*64
        d2 = self.decoder2(d3) + e1  # 4*64*128*128

        project_input = d2.clone()  # 4*64*128*128
        project_output = self.projector(project_input)  # 4*64*128*128

        d1 = self.decoder1(d2) # 4*64*256*256 #注意d_ske的添加和消除
        out = self.finaldeconv1(d1)  # 4*32*512*512
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = nn.Dropout2d(0.5)(out)
        out = self.finalconv3(out)  # 4*2*512*512

        # Skeleton Decoder
        ske_d2 = self.ske_decoder2(d3) + e1 # 4*64*128*128 # 注意e1的增减
        ske_d1 = self.ske_decoder1(ske_d2)  # 4*64*256*256
        ske_out = self.ske_finaldeconv1(ske_d1)  # 4*32*512*512
        ske_out = self.ske_finalrelu1(ske_out)
        ske_out = self.ske_finalconv2(ske_out)
        ske_out = self.ske_finalrelu2(ske_out)
        ske_out = nn.Dropout2d(0.5)(ske_out)
        ske_out = self.finalconv3(ske_out)  # 4*2*512*512

        # 与上面skeleton解码器二选一
        # e1_ske = F.interpolate(e1, size=(256, 256), mode='bilinear', align_corners=False) # 4*64*256*256
        # e2_ske = F.interpolate(e2, size=(256, 256), mode='bilinear', align_corners=False) # 4*128*256*256
        # e3_ske = F.interpolate(e3, size=(256, 256), mode='bilinear', align_corners=False) # 4*256*256*256
        # d4_ske = F.interpolate(d4, size=(256, 256), mode='bilinear', align_corners=False)  # 4*256*256*256
        # d_ske = torch.cat([e1_ske, e2_ske, e3_ske + d4_ske], dim=1) # 4*(64 + 128 + 256) * 256 * 256 = 4 * 448 * 256 * 256
        # d_ske = self.ske_conv1(d_ske) # 4 * 64 * 256 * 256
        # ske_out = self.ske_finaldeconv1(d_ske) # 4 * 32 * 512 * 512
        # ske_out = self.ske_finalrelu1(ske_out)
        # ske_out = self.ske_finalconv2(ske_out)
        # ske_out = self.ske_finalrelu2(ske_out)
        # ske_out = nn.Dropout2d(0.5)(ske_out)
        # ske_out = self.ske_finalconv3(ske_out)

        return out, ske_out, project_output


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = bn
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        y_1 = self.conv(x)
        if self.bn:
            y_1 = self.bn(y_1)
        y_1 = self.relu(y_1)
        return y_1

class Projectors(nn.Module):
    def __init__(self, in_channels=64, out_channels=32):
        super(Projectors, self).__init__()
        self.conv_deconv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=True),
            # Conv(in_channels // 2, out_channels, kernel_size=1, stride=1, padding=0, bn=False),#(32,64,64)
        )

    def forward(self, input): #(128,64,64)
        x_out = self.conv_deconv(input) #(32,64,64)
        return x_out  #(32,64,64)


if __name__=='__main__':
    net=DLinkNet()
    x=torch.randn(1,3,512,512)
    y=net(x)
    print(y.shape)


