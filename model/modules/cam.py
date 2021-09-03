from typing import Tuple, Union

import torch
from torch import nn
import torch.nn.functional as func

from model.manager import Register, Registers

"""
This file contains the modules of Feature Fusion Module from CVPR 2021
Paper will be found at ```https://openaccess.thecvf.com/content/WACV2021/papers/Dai_Attentional_Feature_Fusion_WACV_2021_paper.pdf```

"""


class MSCAMFusion(nn.Module):
    """
    This is the minimum part of Feature Fusion.
    """

    def __init__(self, input_feature_size: Union[int, Tuple], in_channel: int, r: int = 4) -> None:
        """
        Generate the minimum part of Feature Fusion.
        :param input_feature_size: tuple or int, the size of input feature, note that tuple should has two dimensions if given.
        :param in_channel: int, the channels of input feature.
        :param r: int, rescale aspect, should be smaller than in_channel and set to 4 by default.
        """
        super().__init__()
        if isinstance(input_feature_size, int):
            input_feature_size = (input_feature_size, input_feature_size)
        self.global_average_pooling = nn.AvgPool2d(input_feature_size)
        self.point_wise_conv = nn.Conv2d(in_channel, in_channel // r, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(in_channel // r)
        self.point_wise_conv_upsample = nn.Conv2d(in_channel // r, in_channel, kernel_size=(1, 1))
        self.bn_upsample = nn.BatchNorm2d(in_channel)

        self.point_wise_conv_original = nn.Conv2d(in_channel, in_channel // r, kernel_size=(1, 1))
        self.bn_original = nn.BatchNorm2d(in_channel // r)
        self.point_wise_conv_upsample_original = nn.Conv2d(in_channel // r, in_channel, kernel_size=(1, 1))
        self.bn_original_upsample = nn.BatchNorm2d(in_channel)

    def forward(self, input_feature):
        left = self.global_average_pooling(input_feature)
        left = self.point_wise_conv(left)
        left = self.bn(left)
        left = func.relu(left, inplace=True)
        left = self.point_wise_conv_upsample(left)
        left = self.bn_upsample(left)

        right = self.point_wise_conv_original(input_feature)
        right = self.bn_original(right)
        right = func.relu(right, True)
        right = self.point_wise_conv_upsample_original(right)
        right = self.bn_original_upsample(right)

        added = left + right
        sigmoid = func.sigmoid(added)

        short_cut = input_feature * sigmoid
        return short_cut


@Registers.module.register
class IAFFFusion(nn.Module):
    """
    This is the model of Feature Fusion.
    """

    def __init__(self, input_feature_size: Union[int, Tuple], input_channels: int, iteration: int):
        """
        Generate the basic unit of Feature Fusion.
        :param input_feature_size: tuple or int, the size of input feature, note that tuple should has two dimensions if given.
        :param input_channels: int, the channels of input feature.
        :param iteration: int, the iteration times for feature fusion.
        """
        super().__init__()
        cam_list = [MSCAMFusion(input_feature_size, input_channels) for i in range(iteration)]
        self.models = nn.ModuleList(cam_list)

    def forward(self, feature_1, feature_2):
        for module in self.models:
            add_feature = feature_1 + feature_2
            cam_output = module(add_feature)
            feature_1 = cam_output * feature_1
            feature_2 = (1 - cam_output) * feature_2
        return feature_1 + feature_2


if __name__ == '__main__':
    import torchinfo

    model = IAFFFusion((16, 16), 512, 4)
    model = model.cuda()
    torchinfo.summary(model, ((4, 512, 16, 16), (4, 512, 16, 16)))
