import torch
import numpy as np
from torch import nn,einsum
from torchvision import models
from einops import rearrange,repeat
from torch.nn import init
from torch.nn import functional as F
from functools import partial

class CyclicShift(nn.Module):
    def __init__(self,displacement):
        super().__init__()
        self.displacement=displacement
    def forward(self,x):
        return torch.roll(x,shifts=(self.displacement,self.displacement),dims=(1,2))

class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(x,**kwargs)+x

class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,dim),
        )
    def forward(self,x):
        return self.net(x)

def create_mask(window_size,displacement,upper_lower,left_right):
    mask=torch.zeros(window_size**2,window_size**2)
    if upper_lower:
        mask[-displacement*window_size:,:-displacement*window_size]=float('inf')
        mask[:-displacement*window_size,-displacement*window_size:]=float('inf')
    if left_right:
        mask=rearrange(mask,'(h1 w1) (h2 w2)->h1 w1 h2 w2',h1=window_size,h2=window_size)
        mask[:,-displacement:,:,:-displacement]=float('-inf')
        mask[:,:-displacement,:,-displacement:]=float('-inf')
        mask=rearrange(mask,'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
    return mask

def get_relative_distances(window_size):
    indices=torch.tensor(np.array([[x,y] for x in range(window_size) for y in range(window_size)]))
    distances=indices[None,:,:]-indices[:,None,:]
    return distances

class WindowAttention(nn.Module):
    def __init__(self,dim,heads,head_dim,shifted,window_size,relative_pos_embedding):
        super().__init__()
        inner_dim=head_dim*heads
        self.heads=heads
        self.scale=head_dim**(-0.5)
        self.window_size=window_size
        self.relative_pos_embedding=relative_pos_embedding
        self.shifted=shifted

        if self.shifted:
            displacement=window_size//2
            self.cyclic_shift=CyclicShift(-displacement)
            self.cyclic_back_shift=CyclicShift(displacement)
            self.upper_lower_mask=nn.Parameter(
                create_mask(window_size=window_size,displacement=displacement,upper_lower=True,left_right=False,
                requires_grad=False)
            )
            self.left_right_mask=nn.Parameter(
                create_mask(window_size=window_size,displacement=displacement,upper_lower=False,left_right=True,
                requires_grad=False)
            )

        self.to_qkv=nn.Linear(dim,inner_dim*3,bias=False)

        if self.relative_pos_embedding:
            self.relative_indices=get_relative_distances(window_size)+window_size-1
            self.pos_embedding=nn.Parameter(torch.randn(2*window_size-1,2*window_size-1))
        else:
            self.pos_embedding=nn.Parameter(torch.randn(window_size**2,window_size**2))
        self.to_out=nn.Linear(inner_dim,dim)

    def forward(self,x):#(1,128,128,8)
        if self.shifted:
            x=self.cyclic_shift(x)
        b,n_h,n_w,_,h=*x.shape,self.heads#(1,128,128,8),2
        qkv=self.to_qkv(x).chunk(3,dim=-1)#(1,128,128,8*3)->(1,128,128,8)
        # print(qkv[2].shape)
        nw_h=n_h//self.window_size#128//16=8
        nw_w=n_w//self.window_size#128//16=8
        q,k,v=map(
            lambda t:rearrange(t,'b (nw_h w_h) (nw_w w_w) (h d)->b h (nw_h nw_w) (w_h w_w) d',h=h,
                               w_h=self.window_size,w_w=self.window_size),qkv
        )#1,(8,16),(8,16),(2,4)->1,2,(8,8),(16,16),4=1,2,64,256,4
        print(q.shape,k.shape,v.shape)
        dots=einsum('b h w i d, b h w j d -> b h w i j',q,k)*self.scale#1,2,64,256,4 * 1,2,64,256,4
        print(dots.shape)
        if self.relative_pos_embedding:
            dots+=self.pos_embedding[self.relative_indices[:,:,0],self.relative_indices[:,:,1]]
        else:
            dots+=self.pos_embedding
        if self.shifted:
            dots[:,:,-nw_w:]+=self.upper_lower_mask
            dots[:,:,nw_w-1:nw_w]+=self.left_right_mask

        attn=dots.softmax(dim=-1)
        print(attn.shape)
        out=einsum('b h w i j, b h w j d -> b h w i d',attn,v)
        print(out.shape)#1,2,64,256,4

        out=rearrange(out,'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',h=h,
                      w_h=self.window_size,w_w=self.window_size,nw_h=nw_h,nw_w=nw_w) # #1,2,(8,8),(16,16),4->1,(8,16),(8,16),(2,4)
        out=self.to_out(out)
        print(out.shape)
        if self.shifted:
            out=self.cyclic_back_shift(out)

        return out

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

if __name__=='__main__':
    model_p=PatchMerging(3,8,2)
    x=torch.randn(1,3,512,512)
    x_p=model_p(x)
    print(x_p.shape)
    model_s=SwinBlock(dim=8, heads=2, head_dim=4, mlp_dim=4*4, shifted=False, window_size=16, relative_pos_embedding=False)
    x_s=model_s(x_p)
    print(x_s.shape)
