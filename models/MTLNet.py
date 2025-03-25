import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_cbam import BasicBlock
from .pvt import PVT


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True, relu=True):
    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0, bn=True, relu=True):
    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers



class MTLNet(nn.Module):
    def __init__(self, conf = True):
        self.conf = conf
        super(MTLNet, self).__init__()
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1, bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1, bn=False)
        self.conv1 = nn.Sequential(
            conv_bn_relu(64, 64, kernel=3, stride=1, padding=1, bn=False),
            nn.Conv2d(64, 64, kernel_size=1)
        )
        self.backbone = PVT(in_chans=64, patch_size=2, pretrained='./pretrained/pvt.pth',)
        # Shared Decoder
        # 1/16
        self.dec6 = nn.Sequential(
            convt_bn_relu(512, 256, kernel=3, stride=2, padding=1, output_padding=1),
            BasicBlock(256, 256, stride=1, downsample=None, ratio=16),
        )
        # 1/8
        self.dec5 = nn.Sequential(
            convt_bn_relu(256 + 320, 128, kernel=3, stride=2, padding=1, output_padding=1),
            BasicBlock(128, 128, stride=1, downsample=None, ratio=8),
        )
        # Mask Branch
        # 1/4
        self.mask_dec4 = nn.Sequential(
            convt_bn_relu(128 + 128, 64, kernel=3, stride=2, padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )
        # 1/2
        self.mask_dec3 = nn.Sequential(
            convt_bn_relu(64 + 64, 64, kernel=3, stride=2, padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )
        # 1/1
        self.mask_dec2 = nn.Sequential(
            convt_bn_relu(64 + 128, 64, kernel=3, stride=2, padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )

        # 1/1
        self.mask_dec1 = conv_bn_relu(64 + 64, 64, kernel=3, stride=1, padding=1)
        self.mask_dec0 = conv_bn_relu(64 + 64, 1, kernel=3, stride=1, padding=1, bn=False, relu=False)
        self.mask_pred = nn.Sigmoid()
            
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # depth Branch
        # 1/4
        self.depth_dec4 = nn.Sequential(
            convt_bn_relu(128 + 64, 64, kernel=3, stride=2, padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )
        # 1/2
        self.depth_dec3 = nn.Sequential(
            convt_bn_relu(64 + 64, 64, kernel=3, stride=2, padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )
        # 1/1
        self.depth_dec2 = nn.Sequential(
            convt_bn_relu(64 + 64 , 64, kernel=3, stride=2, padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )
        # 1/1
        self.depth_dec1 = conv_bn_relu(64 + 64, 64, kernel=3, stride=1, padding=1)
        self.depth_dec0 = conv_bn_relu(64 + 64, 64, kernel=3, stride=1, padding=1)
        self.depth_init_pred = conv_bn_relu(64 , 1, kernel=3, stride=1, padding=1, bn=False, relu=True)

        self.conf_pred = nn.Sequential(
                conv_bn_relu(64, 32, kernel=3, stride=1, padding=1),
                nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Sigmoid()
        )

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape
        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)
        f = torch.cat((fd, fe), dim=dim)
        return f



    def forward(self, rgb, depth):
        n, h, w = depth.shape
        depth = depth.view(n, 1, h, w)
        depth_raw = depth
        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(depth)
        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
        fe1 = self.conv1(fe1)

        fe2, fe3, fe4, fe5, fe6, fe7 = self.backbone(fe1)
        # Shared Decoding
        fd6 = self.dec6(fe7)
        fd5 = self.dec5(self._concat(fd6, fe6))
        # Mask Decoding
        mask_fd4 = self.mask_dec4(self._concat(fd5, fe5))
        mask_fd3 = self.mask_dec3(self._concat(mask_fd4, fe4))
        mask_fd2 = self.mask_dec2(self._concat(mask_fd3, fe3))
        mask_fd1 = self.mask_dec1(self._concat(mask_fd2, fe2))
        mask_fd0 = self.mask_dec0(self._concat(mask_fd1, fe1))
        mask = self.mask_pred(mask_fd0)
        mask_fd4_down = self.pool(mask_fd4)
        mask_fd3_down = self.pool(mask_fd3)
        mask_fd2_down = self.pool(mask_fd2)
        # depth Decoding
        depth_fd4 = self.depth_dec4(self._concat(fd5,mask_fd4_down))
        depth_fd3 = self.depth_dec3(self._concat(depth_fd4,mask_fd3_down))
        depth_fd2 = self.depth_dec2(self._concat(depth_fd3,mask_fd2_down))
        depth_fd1 = self.depth_dec1(self._concat(depth_fd2,mask_fd1))
        depth_fd0 = self.depth_dec0(self._concat(depth_fd1,mask_fd1))
        depth_init = self.depth_init_pred(depth_fd0 * mask)
        conf = self.conf_pred(depth_fd0)
        if self.conf:
            depth_pred = depth_init * conf + depth_raw * (1-conf)
        else:
            depth_pred = depth_init
            
        return depth_pred,depth_init,conf,mask


# model = MTLNet()
# rgb_input = torch.randn(4, 3, 240, 320)     # [B, 3, H, W]
# depth_input = torch.randn(4, 240, 320)    # [B, H, W]
# a,b,C,D = model(rgb_input,depth_input)
# # for i, feature in enumerate(fused_features):
# #     print(f"Feature {i+1} shape: {feature.shape}")
# print(a.shape )
# print('mask',b.shape )

