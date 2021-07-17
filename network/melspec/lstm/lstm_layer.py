import torch
from torch import nn
import torch.nn.functional as func
import torchinfo
from torch.autograd import Variable


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


class LstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(32, 32, (3, 1), (2, 1))
        self.conv_layer_2 = nn.Conv2d(32, 32, (3, 1), (2, 1))

        self.conv_layer_3 = nn.Conv2d(32, 32, (2, 1), (2, 1))

        self.bilstm_layer_1 = nn.LSTM(input_size=32, hidden_size=30, num_layers=2, bidirectional=True,
                                      batch_first=True)

        self._normal_init(self.conv_layer_1, 0, 0.1)
        self._normal_init(self.conv_layer_2, 0, 0.1)
        self._normal_init(self.conv_layer_3, 0, 0.1)
        for name, param in self.bilstm_layer_1.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        self.hidden = torch.randn((2 * 2, 4, 30)).cuda()
        self.cell = torch.randn((2 * 2, 4, 30)).cuda()

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.conv_layer_1(input_tensor)
        output = self.conv_layer_2(output)
        output = self.conv_layer_3(output)

        length = output.shape[3]
        channel = output.shape[1]
        output = output.permute((0, 3, 1, 2))

        output = output.view(batch_size, length, channel)
        output, (hidden_n, cell_n) = self.bilstm_layer_1(output, (self.hidden, self.cell))
        self.hidden = hidden_n
        self.cell = cell_n

        return output

    @staticmethod
    def _normal_init(m, mean, stddev, truncated=False):
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()


class DenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(900, 64)
        self.dropout = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(64, 3)

        self._normal_init(self.linear_1, 0, 0.01)
        self._normal_init(self.linear_2, 0, 0.01)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = input_tensor.reshape((batch_size, -1))

        output = self.linear_1(output)
        output = func.relu(output, inplace=True)
        output = self.dropout(output)

        output = self.linear_2(output)
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
    torchinfo.summary(LstmModel(), (4, 32, 13, 15))
    torchinfo.summary(DenseModel(), (4, 15, 600))
