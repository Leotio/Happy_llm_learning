'''实现RMSNorm'''
import torch.nn as nn
import torch
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = 