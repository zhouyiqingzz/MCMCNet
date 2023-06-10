import numpy as np
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self,batch_size,image_size,patch_size,in_channels,embed_dim,dropout):
        super(PatchEmbedding,self).__init__()
        n_patchs=(image_size//patch_size)**2
        self.conv1=nn.Conv2d(in_channels,embed_dim,patch_size,patch_size)
        self.dropout=nn.Dropout(dropout)
        self.class_token=torch.randn((batch_size,1,embed_dim))
        self.position=torch.randn((batch_size,n_patchs+1,embed_dim))

    def forward(self,x):
        x=self.conv1(x)
        x=self.flatten(2)
        x=torch.cat((self.class_token,x),axis=1)
        x=x+self.position
        x=self.dropout(x)

        return x

class CalculateAttention(nn.Module):
    def __init__(self):
        super(CalculateAttention,self).__init__()
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,Q,K,V,qkv):
        score=torch.matmul(Q,K.transpose(2,3))/(np.sqrt(qkv))
        score=self.softmax(score)
        score=torch.matmul(score,V)

        return score

class Attention(nn.Module):
    def __init__(self,batch_size,embed_dim,num_heads):
        super(Attention,self).__init__()
        self.qkv=embed_dim//num_heads
        self.batch_size=batch_size
        self.num_heads=num_heads
        self.W_Q=nn.Linear(embed_dim,embed_dim)
        self.W_K=nn.Linear(embed_dim,embed_dim)
        self.W_V=nn.Linear(embed_dim,embed_dim)
        self.cal_attn=Attention()

    def forward(self,x):
        Q=self.W_Q(x).view(self.batch_size,-1,self.num_heads,self.qkv).transpose(1,2)#(batch_size,num_heads,pathcs,qkv)
        K=self.W_K.view(self.batch_size,-1,self.num_heads,self.qkv).transpose(1,2)#(batch_size,num_heads,patchs,qkv)
        V=self.W_V.view(self.batch_size,-1,self.num_heads,self.qkv).transpose(1,2)#(batch_size,num_heads,patchs,qkv)
        att_result=self.cal_attn(Q,K,V,self.qkv)
        att_result=att_result.transpose(1,2).flatten(2)

        return att_result

class EncoderLayer(nn.Module):
    def __init__(self,batch_size,embed_dim,num_heads,mlp_ratio,dropout):
        super(EncoderLayer,self).__init__()
        self.attn_norm=nn.LayerNorm(embed_dim,eps=1e-6)
        self.attn=Attention(batch_size,embed_dim,num_heads)
        self.mlp_norm=nn.LayerNorm(embed_dim,eps=1e-6)
        self.mlp=Mlp(embed_dim,mlp_ratio,dropout)

    def forward(self,x):
        h=x
        x=self.attn_norm(x)
        x=x+h
        h=x
        x=self.mlp_norm(x)
        x=self.mlp(x)
        x=x+h
        return x

class Encoder(nn.Module):
    def __init__(self,batch_size,embed_dim,num_heads,mlp_ratio,dropout,depth):
        super(Encoder,self).__init__()
        layer_list=[]
        for i in range(len(depth)):
            encoder_layer=EncoderLayer(batch_size,embed_dim,num_heads,mlp_ratio,dropout)
            layer_list.append(encoder_layer)
        self.layers=nn.Sequential(*layer_list)
        self.norm=nn.LayerNorm(embed_dim)

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        x=self.norm(x)

        return x

class MLP(nn.Module):
    def __init__(self,embed_dim,mlp_ratio,dropout):
        super(MLP,self).__init__()
        self.fc1=nn.Linear(embed_dim,embed_dim*mlp_ratio)
        self.fc2=nn.Linear(embed_dim*mlp_ratio,embed_dim)
        self.actlayer=nn.GELU()
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)

    def forward(self,x):
        x=self.fc1(x)
        x=self.actlayer(x)
        x=self.dropout1(x)
        x=self.fc2(x)
        x=self.dropout2(x)

        return x

class Vit():
    def __init__(self,batch_size,image_size,patch_size,in_channels,embed_dim,depth,num_heads,mlp_ratio,dropout=0):
        super(Vit,self).__init__()
        self.patch_embedding=PatchEmbedding(batch_size,image_size,patch_size,in_channels,embed_dim,dropout)
        self.encoder=Encoder(batch_size,embed_dim,num_heads,mlp_ratio,dropout,depth)

    def forward(self,x):
        x=self.patch_embedding(x)
        x=self.encoder(x)

        return x

