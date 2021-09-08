import torch
import torch.nn as nn
import torchvision
from torch.utils import model_zoo

from model.manager import Registers

"""
This file contains the modules of Resnet
Paper will be found at ```https://arxiv.org/abs/1512.03385v1```
"""


class BasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """
    # BasicBlock and BottleNeck block have different output size, we use class attribute expansion to distinct
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # Residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # Shortcut
        self.shortcut = nn.Sequential()

        # The shortcut output dimension is not the same with residual function use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=(1, 1), stride=(stride, stride),
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """
    Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=(stride, stride), kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=(stride, stride), kernel_size=(1, 1),
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


@Registers.module.register
class ResNet(nn.Module):

    def __init__(self, num_layers: int, input_channels: int = 1) -> None:
        super().__init__()

        if num_layers not in [18, 34, 50, 101, 152]:
            raise Exception("ResNet is designed to perform 18,34,50,101,152 layers.")
        num_classes = 3
        if num_layers == 18:
            block = BasicBlock
            num_block = [2, 2, 2, 2]
        elif num_layers == 34:
            block = BasicBlock
            num_block = [3, 4, 6, 3]
        elif num_layers == 50:
            block = BottleNeck
            num_block = [3, 4, 6, 3]
        elif num_layers == 101:
            block = BottleNeck
            num_block = [3, 4, 23, 3]
        elif num_layers == 152:
            block = BottleNeck
            num_block = [3, 8, 36, 3]
        else:
            raise Exception("ResNet is designed to perform 18,34,50,101,152 layers.")
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron network layer, ex. conv layer), one layer may
        contain more than one residual block
        :param block: block type, basic block or bottle neck block
        :param out_channels: output depth channel number of this layer
        :param num_blocks: how many blocks per layer
        :param stride: the stride of the first block of this layer
        :return return a resnet layer
        """

        # we have num_block blocks per layer, the first block could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)

        return output


@Registers.module.register
class ResNetBackbone(nn.Module):
    def __init__(self, num_layers=18):
        super().__init__()
        if num_layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError("Layers should be in 18,34,50,101,152")
        model = None
        if num_layers == 18:
            model = torchvision.models.resnet18()
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
            model.load_state_dict(state_dict, strict=False)
        elif num_layers == 34:
            model = torchvision.models.resnet34()
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
            model.load_state_dict(state_dict, strict=False)
        elif num_layers == 50:
            model = torchvision.models.resnet50()
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
            model.load_state_dict(state_dict, strict=False)
        elif num_layers == 101:
            model = torchvision.models.resnet101()
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = torchvision.models.resnet152()
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet152-b121ed2d.pth')
            model.load_state_dict(state_dict, strict=False)
        backbone = list(
            [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4]
        )
        self.model = nn.Sequential(*backbone)

    def forward(self, input_tensor: torch.Tensor):
        return self.model(input_tensor)


if __name__ == '__main__':
    import torchinfo

    models = ResNet(50)
    models.cuda(0)
    torchinfo.summary(models, (4, 1, 128, 157))
