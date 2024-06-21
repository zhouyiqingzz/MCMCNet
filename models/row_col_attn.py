import torch
import torch.nn as nn
from models.base_modules import Conv
import torch.nn.functional as F

class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, shape, dtype=torch.float32))

    def forward(self, x):
        B, C, N = x.shape
        # print(self.pos_embed.shape, x.shape)
        x = x + F.interpolate(self.pos_embed, size=[N], mode='linear', align_corners=False)
        return x

class RowAttn(nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv(dim, nh_kd, kernel_size=1, stride=1, padding=0)
        self.to_k = Conv(dim, nh_kd, kernel_size=1, stride=1, padding=0)
        self.to_v = Conv(dim, self.dh, kernel_size=1, stride=1, padding=0)

        self.proj = nn.Sequential(nn.ReLU(), Conv(self.dh, dim, kernel_size=1, stride=1, padding=0))
        self.proj_encode_row = nn.Sequential(nn.ReLU(), Conv(self.dh, self.dh, kernel_size=1, stride=1, padding=0))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)

        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        xx_row = self.proj(xx_row)
        return xx_row


class ColAttn(nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv(dim, nh_kd, kernel_size=1, stride=1, padding=0)
        self.to_k = Conv(dim, nh_kd, kernel_size=1, stride=1, padding=0)
        self.to_v = Conv(dim, self.dh, kernel_size=1, stride=1, padding=0)

        self.proj = nn.Sequential(nn.ReLU(), Conv(self.dh, dim, kernel_size=1, stride=1, padding=0))
        self.proj_encode_column = nn.Sequential(nn.ReLU(), Conv(self.dh, self.dh, kernel_size=1, stride=1, padding=0))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx_column = self.proj(xx_column)
        return xx_column
