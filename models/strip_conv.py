import torch
import torch.nn as nn
from models.base_modules import Conv,BNPReLU
import torch.nn.functional as F
from models.Sea_Attn import Sea_Attention
from models.AFF_Attn import AFF
torch.manual_seed(1)
#条状卷积解码块
class StripConvBlock(nn.Module):
    def __init__(self, in_channels, n_filters, inp=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(in_channels // 4, in_channels // 4, (1, 9), padding=(0, 4))
        self.deconv2 = nn.Conv2d(in_channels // 4, in_channels // 4, (9, 1), padding=(4, 0))
        self.deconv3 = nn.Conv2d(in_channels // 4, in_channels // 4, (9, 1), padding=(4, 0))
        self.deconv4 = nn.Conv2d(in_channels // 4, in_channels // 4, (1, 9), padding=(0, 4))

        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels, n_filters, 1)
        self.bn3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU()

        self.sea_attn1 = Sea_Attention(dim=in_channels // 4, key_dim=in_channels // 4)
        self.sea_attn2 = Sea_Attention(dim=in_channels // 4, key_dim=in_channels // 4)
        self.sea_attn3 = Sea_Attention(dim=in_channels // 4, key_dim=in_channels // 4)
        self.sea_attn4 = Sea_Attention(dim=in_channels // 4, key_dim=in_channels // 4)
        self.aff_attn = AFF(channels=n_filters)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x1 = self.sea_attn1(x1)
        x2 = self.deconv2(x)
        x2 = self.sea_attn2(x2)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x3 = self.sea_attn3(x3)
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x4 = self.sea_attn4(x4)
        x = torch.cat((x1, x2, x3, x4), 1)
        # if self.inp:
        #     x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.aff_attn(x, res)

        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))#两个参数时填充左右两边
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)



if __name__=='__main__':
    strip_conv=StripConvBlock(32, 32)
    x = torch.randn((1,32,64,64))
    output=strip_conv(x)
    y = F.pad(x, (1, 2))
    print(output.shape, y.shape)
    # row_attn = RowAttn(dim=32, key_dim=32, num_heads=2)
    # col_attn = ColAttn(dim=32, key_dim=32, num_heads=2)
    # x = torch.randn((1,32,64,64))
    # row_output = row_attn(x)
    # col_output = col_attn(x)
    # print(row_output.shape, col_output.shape)
    # output = row_output + col_output
    # print(output.shape)
