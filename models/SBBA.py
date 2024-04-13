from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class SBBA(nn.Module):
    # Shadow Boundary-Aware Attention
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.patch_embed =nn.Conv2d(dim, dim, kernel_size=1, stride=1)

        self.num_heads = heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.heads = heads
        self.scale = dim ** -0.5
        self.inner_dim = self.head_dim * self.heads


        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, pseudo_mask, boundary_mask):
        b,c,h,w = x.shape
        resized_boundary = F.interpolate(boundary_mask, size=(h,w), mode='bilinear', align_corners=False)
        resized_pseudo = F.interpolate(pseudo_mask, size=(h,w), mode='bilinear', align_corners=False)

        resized_pseudo = resized_pseudo.flatten(2).transpose(1, 2)
        x = resized_boundary + self.patch_embed(x) # eq.(6)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        qkv = (self.to_q(x), self.to_k(resized_pseudo*x),self.to_v(resized_pseudo*x))

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out)
        return rearrange(out, 'b (h w) c -> b c h w',h=h,w=w)


if __name__ == '__main__':
    x = torch.randn(4, 128, 64, 64)
    pseudo = torch.randn(4, 1, 512, 512)
    boundary = torch.randn(4, 1, 512, 512)
    import numpy as np
    def cal_param(model):
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in model.parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad:
                Trainable_params += mulValue
            else:
                NonTrainable_params += mulValue

        print(f'Total params: {Total_params / 1e6}M')
        print(f'Trainable params: {Trainable_params / 1e6}M')
        print(f'Non-trainable params: {NonTrainable_params / 1e6}M')

    att = SBBA(dim=512,heads=8)
    #out = att(x,pseudo,boundary)
    print("pause")
    cal_param(att)