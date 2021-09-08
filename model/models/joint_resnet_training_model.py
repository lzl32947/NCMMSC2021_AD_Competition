from typing import Tuple

from torch import nn
import torch
import torch.nn.functional as func
import torchvision
from model.base_model import BaseModel
from model.manager import Register, Registers
# from model.modules.resnet import ResNet
from torch.autograd import Variable


@Registers.model.register
class SpecificTrainResNetModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super(SpecificTrainResNetModel, self).__init__()
        # self.extractor = torchvision.models.resnet18(pretrained=True)
        # self.extractor = ResNet(18)
        self.extractor = Registers.module["ResNet"](18)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.extractor(input_tensor)
        output = self.avg_pool(output)
        output = output.view(batch_size, -1)
        output = self.fc(output)
        return output


@Registers.model.register
class SpecificTrainResNetLongLSTMModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super(SpecificTrainResNetLongLSTMModel, self).__init__()
        # self.extractor = ResNet(18)
        self.extractor = Registers.module["ResNet"](18)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(512, 1024, (1, 1), stride=(1, 1)),
        #     # nn.ReLU(),
        #     # nn.AvgPool2d((1, None))
        #
        # )
        self.layer_dim = 2
        self.hidden_dim = 600
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.hidden_dim, num_layers=self.layer_dim, bidirectional=True
                            , batch_first=True)
        self.fc = nn.Linear(self.hidden_dim * 2, 3)
        # self.fc = nn.Linear(2401, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.extractor(input_tensor)
        output = self.avg_pool(output)
        # output = self.conv_layers(output)

        length = output.shape[3]
        channel = output.shape[1]
        output = output.permute((0, 3, 1, 2))

        output = output.view(batch_size, length, channel)
        h0 = Variable(torch.zeros(self.layer_dim * 2, batch_size, self.hidden_dim).cuda())
        c0 = Variable(torch.zeros(self.layer_dim * 2, batch_size, self.hidden_dim).cuda())
        lstm_out, (h_n, c_n) = self.lstm(output, (h0, c0))

        # output = self.fc(output[:, -1, :])
        # output = output.squeeze(2).permute([2, 0, 1])
        # h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        # c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        # lstm_out, (hn, cn) = self.lstm(output)
        # lstm_out = func.relu(lstm_out)
        # print(lstm_out.shape)
        # lstm_out = lstm_out.contiguous().view(batch_size, -1)
        lstm_out = self.fc(lstm_out[:, -1, :])
        # lstm_out = self.fc(lstm_out)
        return lstm_out


@Registers.model.register
class SpecificTrainResNetLongModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super(SpecificTrainResNetLongModel, self).__init__()
        self.extractor = Registers.module["ResNet"](50)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 6))
        self.fc = nn.Linear(2048 * 6, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.extractor(input_tensor)
        output = self.avg_pool(output)
        long_out = output.view(batch_size, -1)
        long_out = self.fc(long_out)
        long_out2 = func.relu(long_out)
        long_out2 = self.fc2(long_out2)
        long_out3 = func.relu(long_out2)
        long_out3 = self.fc3(long_out3)
        return long_out3


class ConcatModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(6144, 2048, (3, 3))

        self.conv_layer_2 = nn.Conv2d(2048, 1024, (3, 3))
        self.maxpooling_2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_layer_3 = nn.Conv2d(1024, 1024, (3, 3))
        self.maxpooling_3 = nn.MaxPool2d((2, 2), stride=2)

        self.linear_1 = nn.Linear(1024, 1024)
        self.dropout_1 = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(1024, 3)

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


@Registers.model.register
class MSMJointConcatFineTuneResNetModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super(MSMJointConcatFineTuneResNetModel, self).__init__()
        self.extractor_mfcc = Registers.module["ResNet"](50)
        self.extractor_spec = Registers.module["ResNet"](50)
        self.extractor_mel = Registers.module["ResNet"](50)
        self.dense = ConcatModel()
        self.set_expected_input(input_shape)
        self.set_description("MFCC SPEC MELSPEC ResNet Joint 2D Fine-tune Model")

    def forward(self, input_mfcc: torch.Tensor, input_spec: torch.Tensor, input_mel: torch.Tensor):
        output_mfcc = self.extractor_mfcc(input_mfcc)
        output_spec = self.extractor_spec(input_spec)
        output_mel = self.extractor_mel(input_mel)
        concat_output = torch.cat([output_spec, output_mel, output_mfcc], dim=1)
        output = self.dense(concat_output)
        return output


if __name__ == '__main__':
    import torchinfo

    model = SpecificTrainResNetLongLSTMModel(input_shape=())
    model.cuda()
    torchinfo.summary(model, (4, 1, 128, 782))
