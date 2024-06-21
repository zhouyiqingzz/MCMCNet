import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)
class Sea_Attention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=2, attn_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = nn.Conv2d(dim, nh_kd, kernel_size=1, stride=1, padding=0)
        self.to_k = nn.Conv2d(dim, nh_kd, kernel_size=1, stride=1, padding=0)
        self.to_v = nn.Conv2d(dim, self.dh, kernel_size=1, stride=1, padding=0)

        self.proj_encode_row = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.dh, self.dh, kernel_size=1, stride=1, padding=0)
        )
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_col = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.dh, self.dh, kernel_size=1, stride=1, padding=0)
        )
        self.pos_emb_colq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_colk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = nn.Conv2d(2 * nh_kd + self.dh, 2 * self.dh, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()
        self.pwconv = nn.Conv2d(2 * self.dh, dim, kernel_size=1, stride=1, padding=0)

        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.dh, dim, kernel_size=1, stride=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # detail enhancement kernel
        qkv = torch.cat([q, k, v], dim=1)
        # print(qkv.shape)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)

        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        ## squeeze column
        qcol = self.pos_emb_colq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcol = self.pos_emb_colk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcol = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)

        attn_col = torch.matmul(qcol, kcol) * self.scale
        attn_col = attn_col.softmax(dim=-1)
        xx_col = torch.matmul(attn_col, vcol)  # B nH W C
        xx_col = self.proj_encode_col(xx_col.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row + xx_col
        xx = v + xx
        xx = self.proj(xx)
        xx = self.sigmoid(xx) * qkv + xx
        return xx


class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, shape, dtype=torch.float32))

    def forward(self, x):
        B, C, N = x.shape
        # print(self.pos_embed.shape)
        x = x + F.interpolate(self.pos_embed, size=[N], mode='linear', align_corners=False)
        return x



if __name__=='__main__':
    sea_attention = Sea_Attention(dim=256, key_dim=256, num_heads=2)
    x = torch.randn((1,256,32,32))
    output = sea_attention(x)
    print(output.shape)