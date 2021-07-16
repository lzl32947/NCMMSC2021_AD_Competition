import torch
import torchinfo
from torch import nn
import torch.nn.functional as func
from torch.autograd import Variable

class DenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(9000, 64)
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

class lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(96, 96, (3, 1), (2, 1))
        self.conv_layer_2 = nn.Conv2d(96, 96, (3, 1), (2, 1))

        self.conv_layer_3 = nn.Conv2d(96, 96, (2, 1), (2, 1))

        self.bilstm_layer_1 = nn.LSTM(input_size=96, hidden_size=300, num_layers=2, bidirectional=True,
                                      batch_first=True)

        self._normal_init(self.conv_layer_1, 0, 0.1)
        self._normal_init(self.conv_layer_2, 0, 0.1)
        self._normal_init(self.conv_layer_3, 0, 0.1)
        for name, param in self.bilstm_layer_1.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.conv_layer_1(input_tensor)
        output = self.conv_layer_2(output)
        output = self.conv_layer_3(output)


        length = output.shape[3]
        channel = output.shape[1]
        output = output.permute((0, 3, 1, 2))

        output = output.view(batch_size, length, channel)
        hidden = Variable(torch.zeros(2 * 2, batch_size, 300))
        hidden = hidden.cuda()
        cell = Variable(torch.zeros(2 * 2, batch_size, 300))
        cell = cell.cuda()
        output, (hidden_n, cell_n) = self.bilstm_layer_1(output, (hidden, cell))

        return output

    @staticmethod
    def _normal_init(m, mean, stddev, truncated=False):
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(96, 256, (3, 3))

        self.conv_layer_2 = nn.Conv2d(256, 512, (3, 3))
        self.maxpooling_2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_layer_3 = nn.Conv2d(512, 1024, (3, 3))
        self.maxpooling_3 = nn.MaxPool2d((2, 2), stride=2)

        self.linear_1 = nn.Linear(1024, 1024)
        self.dropout_1 = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(1024, 3)

        self._normal_init(self.conv_layer_1, 0, 0.1)
        self._normal_init(self.conv_layer_2, 0, 0.1)
        self._normal_init(self.conv_layer_3, 0, 0.1)
        self._normal_init(self.linear_1, 0, 0.1)
        self._normal_init(self.linear_2, 0, 0.1)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output, inplace=True)

        output = self.conv_layer_2(output)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_2(output)

        output = self.conv_layer_3(output)
        output = func.relu(output, inplace=True)
        output = self.maxpooling_3(output)

        output = output.view((batch_size, -1))
        output = self.linear_1(output)
        output = self.dropout_1(output)
        output = self.linear_2(output)
        return output

    @staticmethod
    def _normal_init(m, mean, stddev, truncated=False):
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()


class GeneralModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.extractor_mfcc = ExtractionModel()
        self.extractor_spec = ExtractionModel()
        self.extractor_mel = ExtractionModel()
        self.lstm = lstm_model()
        self.dense = DenseModel()

    def forward(self, input_mfcc: torch.Tensor, input_spec: torch.Tensor, input_mel: torch.Tensor):
        output_mfcc = self.extractor_mfcc(input_mfcc)
        output_spec = self.extractor_spec(input_spec)
        output_mel = self.extractor_mel(input_mel)
        concat_output = torch.cat([output_spec, output_mel, output_mfcc], dim=1)
        output_lstm = self.lstm(concat_output)
        output = self.dense(output_lstm)
        return output

if __name__ == '__main__':
    torchinfo.summary(ExtractionModel(), (4, 1, 513, 157))
    torchinfo.summary(lstm_model(), (4, 32, 61, 15))
    torchinfo.summary(DenseModel(), (4, 15, 96))
