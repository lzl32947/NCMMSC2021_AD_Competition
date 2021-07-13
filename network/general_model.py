import torch
from torch import nn
import torch.nn.functional as func
import torchinfo


class MelSpecsExtractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(1, 32, (3, 3))
        self.maxpooling_1 = nn.MaxPool2d((2, 2), stride=(2, 1))

        self.conv_layer_2 = nn.Conv2d(32, 128, (3, 3))
        self.maxpooling_2 = nn.MaxPool2d((2, 2), stride=(2, 1))

        self.conv_layer_3 = nn.Conv2d(128, 512, (3, 3))
        self.maxpooling_3 = nn.MaxPool2d((2, 2), stride=(2, 1))

        self.conv_layer_4 = nn.Conv2d(512, 1024, (3, 3))

        self._normal_init(self.conv_layer_1, 0, 0.1)
        self._normal_init(self.conv_layer_2, 0, 0.1)
        self._normal_init(self.conv_layer_3, 0, 0.1)
        self._normal_init(self.conv_layer_4, 0, 0.1)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_1(output)

        output = self.conv_layer_2(output)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_2(output)

        output = self.conv_layer_3(output)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_3(output)

        output = self.conv_layer_4(output)
        output = func.relu(output, inplace=True)

        return output

    @staticmethod
    def _normal_init(m, mean, stddev, truncated=False):
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()


class SpecsExtractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.maxpooling_1 = nn.MaxPool2d((4, 2), stride=(2, 1), padding=(1, 0))

        self.conv_layer_2 = nn.Conv2d(32, 128, (3, 3), padding=1)
        self.maxpooling_2 = nn.MaxPool2d((4, 2), stride=(2, 1), padding=1)

        self.conv_layer_3 = nn.Conv2d(128, 512, (3, 3), padding=1)
        self.maxpooling_3 = nn.MaxPool2d((4, 2), stride=(2, 1), padding=(1, 0))

        self.conv_layer_4 = nn.Conv2d(512, 1024, (3, 3), padding=1)
        self.maxpooling_4 = nn.MaxPool2d((4, 2), stride=(2, 1), padding=1)

        self.conv_layer_5 = nn.Conv2d(1024, 1024, (3, 3), padding=(1, 0))
        self.maxpooling_5 = nn.MaxPool2d((4, 2), stride=(2, 1), padding=1)

        self._normal_init(self.conv_layer_1, 0, 0.1)
        self._normal_init(self.conv_layer_2, 0, 0.1)
        self._normal_init(self.conv_layer_3, 0, 0.1)
        self._normal_init(self.conv_layer_4, 0, 0.1)
        self._normal_init(self.conv_layer_5, 0, 0.1)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_1(output)

        output = self.conv_layer_2(output)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_2(output)

        output = self.conv_layer_3(output)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_3(output)

        output = self.conv_layer_4(output)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_4(output)

        output = self.conv_layer_5(output)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_5(output)
        return output

    @staticmethod
    def _normal_init(m, mean, stddev, truncated=False):
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()


class MFCCExtractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(1, 32, (3, 3), padding=(0, 1))
        self.conv_layer_2 = nn.Conv2d(32, 128, (3, 3), padding=(0, 1))
        self.conv_layer_3 = nn.Conv2d(128, 512, (3, 3), padding=1)
        self.conv_layer_4 = nn.Conv2d(512, 1024, (3, 3), padding=1)

        self._normal_init(self.conv_layer_1, 0, 0.1)
        self._normal_init(self.conv_layer_2, 0, 0.1)
        self._normal_init(self.conv_layer_3, 0, 0.1)
        self._normal_init(self.conv_layer_4, 0, 0.1)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output, inplace=True)

        output = self.conv_layer_2(output)
        output = func.relu(output, inplace=True)

        output = self.conv_layer_3(output)
        output = func.relu(output, inplace=True)

        output = self.conv_layer_4(output)
        output = func.relu(output, inplace=True)

        return output

    @staticmethod
    def _normal_init(m, mean, stddev, truncated=False):
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()


if __name__ == '__main__':
    # torchinfo.summary(SpecsExtractionModel().cuda(), (4, 1, 513, 157))
    # torchinfo.summary(MelSpecsExtractionModel().cuda(), (4, 1, 128, 157))
    torchinfo.summary(MFCCExtractionModel().cuda(), (4, 1, 20, 157))
