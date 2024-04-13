import torch
from torch import nn
import math
import torch.nn.functional as F


class Fuse1(nn.Module):
    def __init__(self, indim, outdim=None):
        super(Fuse1, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outdim)

    def forward(self, cur, others):

        x = torch.cat([cur, others], dim=1)
        r = self.conv1(F.relu(x))
        r = self.bn1(r)
        r = self.relu(r)
        r = self.conv2(r)
        r = self.bn2(r)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + r)

class Fuse2(nn.Module):
    def __init__(self):
        super(Fuse2, self).__init__()
    def forward(self, cur, others):
        temp = cur+ others
        return temp


class DSA(nn.Module):
    def __init__(self):
        super(DSA, self).__init__()

        '''
        Input: Aligned frames, [B, T, C, H, W]
        Output: Fused frame, [B, C, H, W]
        '''

        self.short_fuse = Fuse2()
        self.long_fuse = Fuse1(1024, 512)
        #self.long_fuse = Fuse2()
        self.integrate_fuse = Fuse1(1536, 512)

    def short_aff(self, cur_fea, short_fea):
        B, T, C, H, W = short_fea.shape

        anchor_fea = cur_fea.flatten(start_dim=2)

        last_next_fea = torch.cat([short_fea[:, [0], ...], short_fea[:, [2], ...]], dim=1)  # b,2,c,h,w
        last_next_fea = last_next_fea.transpose(1, 2).contiguous()
        last_next_fea = last_next_fea.view(B, C, 2 * H * W)  # here should be T-1=2

        # apply fast l2 similarity, follow STCN
        a_sq = last_next_fea.pow(2).sum(1).unsqueeze(2)  # 4 576 1, where 1 is expand by unsqueeze
        ab = last_next_fea.transpose(1, 2) @ anchor_fea
        affinity = (2 * ab - a_sq) / math.sqrt(C)  # B, THW, HW

        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        short_read = torch.bmm(last_next_fea, affinity)
        return short_read.view(B, C, H, W)

    def long_aff(self, cur_fea, long_fea):
        B, T, C, H, W = long_fea.shape

        anchor_fea = cur_fea.clone()
        anchor_fea = anchor_fea.flatten(start_dim=2)  # b,c,h*w

        long_fea = long_fea.transpose(1, 2).contiguous()  # B,C,T,H,W
        long_fea = long_fea.view(B, C, T * H * W)

        # long-term
        a_sq = long_fea.pow(2).sum(1).unsqueeze(2)  # 4 576 1, where 1 is expand by unsqueeze
        ab = long_fea.transpose(1, 2) @ anchor_fea
        affinity = (2 * ab - a_sq) / math.sqrt(C)  # B, THW, HW
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        long_affinity = x_exp / x_exp_sum
        long_affinity = long_affinity.view(B, T, H * W, H * W)

        # self-term
        a_sq = anchor_fea.pow(2).sum(1).unsqueeze(2)  # 4 576 1, where 1 is expand by unsqueeze
        ab = anchor_fea.transpose(1, 2) @ anchor_fea
        affinity = (2 * ab - a_sq) / math.sqrt(C)
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        self_affinity = x_exp / x_exp_sum
        self_affinity = self_affinity.view(B, 1, H * W, H * W)

        residual_affinity = torch.abs(long_affinity - self_affinity).view(B, T * H * W, H * W)
        long_read = torch.bmm(long_fea, residual_affinity)
        return long_read.view(B, C, H, W)

    def forward(self, features):
        b, t, c, h, w = features.size()  # feature is the top one, i.e., b,t,520,16,16
        for idx in range(t):
            cur_fea = features[:, idx, ...]  # b,c,h,w, omit time channel
            if idx == 0:  # if first frame or last frame
                short_range = [idx, idx, idx + 1]
                long_range = [i for i in range(t) if i not in short_range]

            elif idx == (t - 1):
                short_range = [idx - 1, idx, idx]
                long_range = [i for i in range(t) if i not in short_range]

            else:
                short_range = [idx - 1, idx, idx + 1]
                long_range = [i for i in range(t) if i not in short_range]

            short_fea = features[:, short_range, ...]
            long_fea = features[:, long_range, ...]

            short_visit = self.short_aff(cur_fea, short_fea)
            long_visit = self.long_aff(cur_fea, long_fea)

            short_fuse = self.short_fuse(cur_fea, short_visit) # we find use pixel-wise adding performs better
            long_fuse = self.long_fuse(cur_fea, long_visit) # # we find use residual concat performs better, or the loss will be NAN

            fusion = self.integrate_fuse(cur_fea,torch.cat([short_fuse,long_fuse],dim=1)) # we find use residual concat performs better
            # fusion = self.integrate_fuse(short_fuse,long_fuse)
            #fusion = short_fuse+long_fuse
            features[:, idx, ...] = fusion

        return features


if __name__ == '__main__':
    model = DSA()
    def cal_param(model):
        import numpy as np
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
    cal_param(model)