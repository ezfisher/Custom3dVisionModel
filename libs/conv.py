import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super().__init__()
        self.__weight_dh = torch.nn.Parameter(torch.rand((out_channels, in_channels, kernel_size[0], kernel_size[1])))
        self.__weight_dw = torch.nn.Parameter(torch.rand((out_channels, in_channels, kernel_size[0], kernel_size[2])))
        self.__weight_hw = torch.nn.Parameter(torch.rand((out_channels, in_channels, kernel_size[1], kernel_size[2])))
        self.weight = nn.ParameterList([self.__weight_dh, self.__weight_dw, self.__weight_hw])
        self.bias = torch.nn.Parameter(torch.rand(out_channels)) if bias else None
        self.__padding_dh = (padding[0], padding[1])
        self.__padding_dw = (padding[0], padding[2])
        self.__padding_hw = (padding[1], padding[2])
        self.__stride_dh = (stride[0], stride[1])
        self.__stride_dw = (stride[0], stride[2])
        self.__stride_hw = (stride[1], stride[2])
        self.__padding_mode = 'zeros'
        self.__dilation = dilation
        self.__groups = groups
    
    def forward(self, X):
        outdh, outdw, outhw = X.sum(dim=-1), X.sum(dim=-2), X.sum(dim=-3)
        outdh = F.conv2d(outdh, self.__weight_dh, self.bias, self.__stride_dh, self.__padding_dh, self.__dilation, self.__groups)
        outdw = F.conv2d(outdw, self.__weight_dw, self.bias, self.__stride_dw, self.__padding_dw, self.__dilation, self.__groups)
        outhw = F.conv2d(outhw, self.__weight_hw, self.bias, self.__stride_hw, self.__padding_hw, self.__dilation, self.__groups)
        out = outdh.unsqueeze(-1) + outdw.unsqueeze(-2) + outhw.unsqueeze(-3)
        return out