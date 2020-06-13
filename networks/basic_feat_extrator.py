import torch
import torch.nn as nn
import torch.nn.functional as F
from core_dl.base_net import BaseNet
import shutil
import os

# Internal libs
import networks.backbone_drn as drn


class RGBNet(nn.Module):
    def __init__(self, input_dim=(3, 256, 256)):
        """
        Network for extracting RGB features
        :param input_dim: size of the input image
        """
        super(RGBNet, self).__init__()
        self.input_shape_chw = input_dim

        drn_module = drn.drn_d_38(pretrained=False)  # use DRN22 for now
        self.block0 = nn.Sequential(
            drn_module.layer0,
            drn_module.layer1
        )
        self.block1 = drn_module.layer2
        self.block2 = drn_module.layer3
        self.block3 = drn_module.layer4
        self.block4 = drn_module.layer5
        self.block5 = drn_module.layer6
        self.block6 = nn.Sequential(
            drn_module.layer7,
            drn_module.layer8
        )

    def forward(self, x):
        """
        forward with image & scene feature
        :param image: (N, C, H, W)
        :return:
        """
        x0 = self.block0(x)     # 256
        x1 = self.block1(x0)    # 128
        x2 = self.block2(x1)    # 64
        x3 = self.block3(x2)    # 32
        x4 = self.block4(x3)    # 16
        x5 = self.block5(x4)    # 8
        x6 = self.block6(x5)    # 4
        return x6, x5, x4, x3, x2, x1, x0


if __name__ == '__main__':
    from core_dl.module_util import summary_layers
    model = RGBNet().cuda()
    summary_layers(model, input_size=(3, 256, 256))
