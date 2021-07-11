import torch
from torch import nn
import torch.nn.functional as func


class ExtractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(1, 16, (3, 3))
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv_layer_2 = nn.Conv2d(16, 32, (3, 3))
        self.batch_norm2 = nn.BatchNorm2d(32)

        self.conv_layer_3 = nn.Conv2d(32, 64, (3, 3))
        self.conv_layer_4 = nn.Conv2d(64, 128, (3, 3))

        self.conv_layer_5 = nn.Conv2d(128, 256, (3, 3))
        self.conv_layer_6 = nn.Conv2d(256, 512, (5, 5))

        self._normal_init(self.conv_layer_1, 0, 0.01)
        self._normal_init(self.conv_layer_2, 0, 0.01)
        self._normal_init(self.conv_layer_3, 0, 0.01)
        self._normal_init(self.conv_layer_4, 0, 0.01)
        self._normal_init(self.conv_layer_5, 0, 0.01)
        self._normal_init(self.conv_layer_6, 0, 0.01)
        self._normal_init(self.batch_norm1, 0, 0.01)
        self._normal_init(self.batch_norm2, 0, 0.01)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output, inplace=True)
        output = self.batch_norm1(output)

        output = self.conv_layer_2(output)
        output = func.relu(output, inplace=True)
        output = self.batch_norm2(output)

        output = nn.MaxPool2d(kernel_size=(4, 2), stride=(2, 1))(output)

        output = self.conv_layer_3(output)
        output = func.relu(output, inplace=True)

        output = nn.MaxPool2d(kernel_size=(4, 2), stride=(2, 2))(output)

        output = self.conv_layer_4(output)
        output = func.relu(output, inplace=True)

        output = nn.MaxPool2d(kernel_size=(4, 2), stride=(2, 2))(output)

        output = self.conv_layer_5(output)
        output = func.relu(output, inplace=True)

        output = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(output)

        output = self.conv_layer_6(output)
        output = func.relu(output, inplace=True)

        output = output.permute((0, 1, 3, 2))
        output = output.squeeze(dim=3)
        return output

    @staticmethod
    def _normal_init(m, mean, stddev, truncated=False):
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()
