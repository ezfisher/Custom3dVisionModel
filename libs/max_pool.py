import torch.nn as nn
import torch.nn.functional as F

class MaxPool3d(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.__kernel_size_dh = (kernel_size[0], kernel_size[1])
        self.__kernel_size_dw = (kernel_size[0], kernel_size[2])
        self.__kernel_size_hw = (kernel_size[1], kernel_size[2])
        self.__stride_dh = (stride[0], stride[1])
        self.__stride_dw = (stride[0], stride[2])
        self.__stride_hw = (stride[1], stride[2])

    def forward(self, X):
        outdh, outdw, outhw = X.sum(dim=-1), X.sum(dim=-2), X.sum(dim=-3)
        outdh = F.max_pool2d(outdh, self.__kernel_size_dh, self.__stride_dh)
        outdw = F.max_pool2d(outdw, self.__kernel_size_dw, self.__stride_dw)
        outhw = F.max_pool2d(outhw, self.__kernel_size_hw, self.__stride_hw)
        out = outdh.unsqueeze(-1) + outdw.unsqueeze(-2) + outhw.unsqueeze(-3)
        return out