import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Function
class SEAttention1d(nn.Module):
    def __init__(self, channel, reduction):
        super(SEAttention1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        b, c, _,_ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.5, eps=1e-5):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self._activated = True
    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})'
    def set_activation_status(self, status=True):
        self._activated = status
    def forward(self, x):
        if not self.training or not self._activated:
            return x
        if random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig
        lmda = self.beta.sample((B, 1, 1, 1))
        #lmda = torch.full((B, 1, 1, 1), 0.5)#固定lambda值是0.5
        lmda = lmda.to(x.device)
        perm = torch.randperm(B)#随机生成一个0~B-1的数字
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)
        x = x_normed*sig_mix + mu_mix
        return x
class macnn_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=None,   # e.g. [3, 5, 7]
        reduction=6
    ):
        super(macnn_block, self).__init__()
        if kernel_size is None:
            kernel_size = [3, 6, 12]#3, 6, 12
        self.dw1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size[0], stride=1,
                             padding='same', groups=in_channels, bias=False)
        self.pw1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dw2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size[1], stride=1,
                             padding='same', groups=in_channels, bias=False)
        self.pw2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dw3 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size[2], stride=1,
                             padding='same', groups=in_channels, bias=False)
        self.pw3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEAttention1d(out_channels * 3, reduction)
    def forward(self, x):
        x1 = self.pw1(self.dw1(x))
        x2 = self.pw2(self.dw2(x))
        x3 = self.pw3(self.dw3(x))
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)
        return out
class Extractor(nn.Module):
    def __init__(self, in_channels=16, channels=4, block_num=None):
        super(Extractor, self).__init__()
        if block_num is None:
            block_num = [2, 2, 2]
        self.in_channel = in_channels
        self.stem = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 11), padding=(1, 5), stride=2, bias=False)
        #self.stem = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=2, bias=False)
        self.channel = channels
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = self._make_layer(macnn_block, block_num[0], self.channel)
        self.layer2 = self._make_layer(macnn_block, block_num[1], self.channel * 2)
        self.layer3 = self._make_layer(macnn_block, block_num[2], self.channel * 4)
        self.mixstyle = MixStyle(p=0.9, alpha=0.9)  #
        self.pro_head = nn.Linear(48, 128)
    def _make_layer(self, block, block_num, channel, reduction=16):
        layers = []
        for i in range(block_num):
            layers.append(block(self.in_channel, channel, kernel_size=None))
            self.in_channel = 3 * channel
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x) #降采样模块
        x = self.layer1(x)
        x = self.max_pool1(x)
        x = self.layer2(x)
        x = self.max_pool2(x)
        x = self.layer3(x)
        x1 = 0
        features1 = 0
        if self.training:
            x1 = self.mixstyle(x)#生成域
            x1 = self.avg_pool(x1)
            x1 = torch.flatten(x1, 1)
            features1 = self.pro_head(x1)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        features = self.pro_head(x)
        return x, x1, features, features1


