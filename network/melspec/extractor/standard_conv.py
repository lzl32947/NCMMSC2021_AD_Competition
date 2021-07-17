import torch
from torch import nn
import torch.nn.functional as func
import torchinfo


class ExtractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(1, 32, (3, 3))
        self.maxpooling_1 = nn.MaxPool2d((2, 4), stride=2)
        self.batchnorm_1 = nn.BatchNorm2d(32)

        self.conv_layer_2 = nn.Conv2d(32, 32, (3, 3))
        self.maxpooling_2 = nn.MaxPool2d((4, 4), stride=2)
        self.batchnorm_2 = nn.BatchNorm2d(32)

        self.conv_layer_3 = nn.Conv2d(32, 32, (3, 3))
        self.maxpooling_3 = nn.MaxPool2d((2, 5), stride=2)
        self.batchnorm_3 = nn.BatchNorm2d(32)

        self._normal_init(self.conv_layer_1, 0, 0.1)
        self._normal_init(self.batchnorm_1, 0, 0.1)
        self._normal_init(self.conv_layer_2, 0, 0.1)
        self._normal_init(self.batchnorm_2, 0, 0.1)
        self._normal_init(self.conv_layer_3, 0, 0.1)
        self._normal_init(self.batchnorm_3, 0, 0.1)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_1(output)
        output = self.batchnorm_1(output)

        output = self.conv_layer_2(output)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_2(output)
        output = self.batchnorm_2(output)

        output = self.conv_layer_3(output)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_3(output)
        output = self.batchnorm_3(output)

        return output

    @staticmethod
    def _normal_init(m, mean, stddev, truncated=False):
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()


if __name__ == '__main__':
    torchinfo.summary(ExtractionModel(), (4, 1, 128, 157))
