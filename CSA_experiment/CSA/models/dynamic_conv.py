import torch
import torch.nn as nn
import torch.nn.functional as F

class attention2d(nn.Module):
    def __init__(self,in_channels,K):
        super(attention2d,self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Conv2d(in_channels,K,1)
        self.fc2=nn.Conv2d(K,K,1)

    def forward(self,x):
        x=self.avgpool(x)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x).view(x.size(0),-1)

        return F.softmax(x,1)

class Dynamic_conv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1,dilation=1,groups=1,
                 bias=True,K=4):
        super(Dynamic_conv2d,self).__init__()
        assert in_channels % groups==0#不满足则报错
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=groups
        self.bias=bias
        self.K=K
        self.attention=attention2d(in_channels,K)
        self.weight=nn.Parameter(torch.Tensor(K,out_channels,in_channels//groups,kernel_size,kernel_size),requires_grad=True)

        if bias:
            self.bias=nn.Parameter(torch.Tensor(K,out_channels))
        else:
            self.bias=None

    def forward(self,x):
        softmax_attention=self.attention(x)
        batch_size,in_channels,height,width=x.size()
        x=x.view(1,-1,height,width)
        weight=self.weight.view(self.K,-1)

        aggregate_weight=torch.mm(softmax_attention,weight).view(-1,self.in_channels,self.kernel_size,self.kernel_size)

        if self.bias is not None:
            aggregate_bias=torch.mm(softmax_attention,self.bias).view(-1)
            output=F.conv2d(x,weight=aggregate_weight,bias=aggregate_bias,stride=self.stride,padding=self.padding,
                            dilation=self.dilation,groups=self.groups*batch_size)
        else:
            output=F.conv2d(x,weight=aggregate_weight,bias=None,stride=self.stride,padding=self.padding,
                            dilation=self.dilation,groups=self.groups*batch_size)

        output=output.view(batch_size,self.out_channels,output.size(-2),output.size(-1))

        return output

if __name__=="__main__":
    x=torch.randn(1,3,256,256)
    model=Dynamic_conv2d(in_channels=3,out_channels=3,kernel_size=3)
    output=model(x)
    print(output.shape)