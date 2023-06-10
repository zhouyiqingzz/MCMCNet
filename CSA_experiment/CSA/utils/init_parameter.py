import torch.nn as nn

#初始化模型权重
def __init_weight(feature,conv_init,norm_layer,bn_eps,bn_momentum,**kwargs):
    for name,m in feature.named_modules():
        if isinstance(m,(nn.Conv2d,nn.Conv3d)):
            conv_init(m.weight,**kwargs)
        elif isinstance(m,norm_layer):
            m.eps=bn_eps
            m.momentum=bn_momentum
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)

def init_weight(module_list,conv_init,norm_layer,bn_eps,bn_momentum,**kwargs):
    if isinstance(module_list,list):
        for feature in module_list:
            __init_weight(feature,conv_init,norm_layer,bn_eps,bn_momentum,**kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)

#计算模型总的参数量
def network_parameters(model):
    total_parameters=0
    for parameter in model.parameters():
        i=len(parameter.size())
        p=1
        for j in range(i):
            p*=parameter.size(j)
        total_parameters+=p
    return total_parameters