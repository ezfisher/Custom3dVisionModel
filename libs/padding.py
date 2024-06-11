import torch
from torch import nn
from typing import Iterable

class Pad3D(nn.Module):
    def __init__(self, padding):
        super().__init__()
        assert len(padding) == 3
        self.padding = self._check_padding(padding)
    
    def _check_padding(self, padding):
        _padding = []
        for i, p in enumerate(padding):
            if not isinstance(p, Iterable):
                _padding.append((p, p))
            else:
                _padding.append(p)
        return tuple(_padding)

    def _pad_depth(self, X):
        dz0 = torch.zeros(X[:, :, 0, :, :].unsqueeze(-3).shape)
        dz1 = torch.zeros(X[:, :, 0, :, :].unsqueeze(-3).shape)
        return torch.concat((dz0, X, dz1), dim=-3)
    
    def _pad_height(self, X):
        dw0 = torch.zeros(X[:, :, :, 0, :].unsqueeze(-2).shape)
        dw1 = torch.zeros(X[:, :, :, 0, :].unsqueeze(-2).shape)
        return torch.concat((dw0, X, dw1), dim=-2)

    def _pad_width(self, X):
        dh0 = torch.zeros(X[:, :, :, :, 0].unsqueeze(-1).shape)
        dh1 = torch.zeros(X[:, :, :, :, 0].unsqueeze(-1).shape)
        return torch.concat((dh0, X, dh1), dim=-1)

    def _pad(self, X):
        out = self._pad_depth(X)
        out = self._pad_height(out)
        out = self._pad_width(out)
        return out

    def forward(self, X):
        return self._pad(X)