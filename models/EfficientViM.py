import math
import torch
import torch.nn as nn
from .utils import LayerNorm3D, ConvLayer1D, ConvLayer3D, FFN3D

class HSMSSD3D(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = nn.Conv1d(d_model, 3 * state_dim, 1)
        self.dw = ConvLayer3D(3 * state_dim, 3 * state_dim, kernel_size=3, padding=1, groups=3 * state_dim, norm=None, act_layer=None, bn_weight_init=0)
        self.hz_proj = nn.Conv1d(d_model, 2 * self.d_inner, 1)
        self.out_proj = nn.Conv1d(self.d_inner, d_model, 1)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        N = D * H * W
        x_flat = x.view(B, C, N)  # (B, C, L)

        BCdt = self.BCdt_proj(x_flat).view(B, -1, D, H, W)  # (B, 3S, D, H, W)
        BCdt = self.dw(BCdt).view(B, -1, N)

        B_, C_, dt = torch.split(BCdt, [self.state_dim] * 3, dim=1)
        A = (dt + self.A.view(1, -1, 1)).softmax(-1)

        AB = A * B_
        h = x_flat @ AB.transpose(-2, -1)

        h, z = torch.split(self.hz_proj(h), [self.d_inner] * 2, dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        y = h @ C_  # B C N

        y = y.view(B, -1, D, H, W)
        return y, h

class EfficientViMBlock3D(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.mixer = HSMSSD3D(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm3D(dim)

        self.dwconv1 = ConvLayer3D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)
        self.dwconv2 = ConvLayer3D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)

        self.ffn = FFN3D(in_dim=dim, dim=int(dim * mlp_ratio))

        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1, 1)

        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)
        x_prev = x
        x, h = self.mixer(self.norm(x))
        x = (1 - alpha[1]) * x_prev + alpha[1] * x
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)
        return x, h




