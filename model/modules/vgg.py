import math
import torch
import torch.nn as nn
import torchvision
from torch.utils import model_zoo

from model.manager import Registers
import torchvision.models

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


@Registers.module.register
class VggNetBackbone(nn.Module):
    def __init__(self, num_layers=19):
        super().__init__()
        if num_layers not in [11, 13, 16, 19, 20]:
            raise RuntimeError("Layers should be in 11, 13, 16, 19, 19_bn")
        model = None
        if num_layers == 11:
            model = torchvision.models.vgg11()
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg11-bbd30ac9.pth')
            model.load_state_dict(state_dict, strict=False)
        elif num_layers == 13:
            model = torchvision.models.vgg13()
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg13-c768596a.pth')
            model.load_state_dict(state_dict, strict=False)
        elif num_layers == 16:
            model = torchvision.models.vgg16()
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
            model.load_state_dict(state_dict, strict=False)
        elif num_layers == 19:
            model = torchvision.models.vgg19()
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pt')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = torchvision.models.vgg19_bn()
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg19_bn-c79401a0.pth')
            model.load_state_dict(state_dict, strict=False)

        backbone = list(
            [model.features]
        )
        self.model = nn.Sequential(*backbone)

    def forward(self, input_tensor: torch.Tensor):
        return self.model(input_tensor)




if __name__ == '__main__':

    print(torchvision.models.vgg19())
    print(torchvision.models.resnet18())

