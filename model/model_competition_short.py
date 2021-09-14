from typing import Tuple

import torchvision.models
from torch import nn
import torch
import torch.nn.functional as func
from torch.utils import model_zoo

from model.base_model import BaseModel
from model.manager import Register, Registers


class DenseModel(nn.Module):
    def __init__(self, input_unit=6240):
        super().__init__()
        self.linear_1 = nn.Linear(input_unit, 64)
        self.dropout = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(64, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = input_tensor.view((batch_size, -1))

        output = self.linear_1(output)
        output = func.relu(output)
        output = self.dropout(output)

        output = self.linear_2(output)
        return output


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

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output)
        output = self.maxpooling_1(output)
        output = self.batchnorm_1(output)

        output = self.conv_layer_2(output)
        output = func.relu(output)
        output = self.maxpooling_2(output)
        output = self.batchnorm_2(output)

        output = self.conv_layer_3(output)
        output = func.relu(output)
        output = self.maxpooling_3(output)
        output = self.batchnorm_3(output)

        return output


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

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output)

        output = self.conv_layer_2(output)
        output = func.relu(output)
        output = self.maxpooling_2(output)

        output = self.conv_layer_3(output)
        output = func.relu(output)
        output = self.maxpooling_3(output)

        output = output.view((batch_size, -1))
        output = self.linear_1(output)
        output = self.dropout_1(output)
        output = self.linear_2(output)
        return output


@Registers.model.register
class CompetitionSpecificTrainModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor = ExtractionModel()
        self.dense = DenseModel()

    def forward(self, input_tensor: torch.Tensor):
        output = self.extractor(input_tensor)
        output = self.dense(output)
        return output


@Registers.model.register
class CompetitionMSMJointConcatFineTuneModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor_mfcc = ExtractionModel()
        self.extractor_spec = ExtractionModel()
        self.extractor_mel = ExtractionModel()
        self.dense = ConcatModel()

    def forward(self, input_mfcc: torch.Tensor, input_spec: torch.Tensor, input_mel: torch.Tensor):
        output_mfcc = self.extractor_mfcc(input_mfcc)
        output_spec = self.extractor_spec(input_spec)
        output_mel = self.extractor_mel(input_mel)
        concat_output = torch.cat([output_spec, output_mel, output_mfcc], dim=1)
        output = self.dense(concat_output)
        return output


@Registers.model.register
class SpecificTrainVggNet19BackboneLongModel(BaseModel):
    def __init__(self):
        super(SpecificTrainVggNet19BackboneLongModel, self).__init__()
        self.extractor = Registers.module["VggNetBackbone"]('19')
        self.fc = nn.Linear(512 * 4 * 4, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.extractor(input_tensor)
        long_out = output.view(batch_size, -1)
        long_out = self.fc(long_out)
        long_out2 = func.relu(long_out)
        long_out2 = self.dropout1(long_out2)
        long_out2 = self.fc2(long_out2)
        long_out3 = func.relu(long_out2)
        long_out3 = self.dropout2(long_out3)
        long_out3 = self.fc3(long_out3)
        long_out4 = func.relu(long_out3)
        long_out4 = self.dropout3(long_out4)
        long_out4 = self.fc4(long_out4)
        return long_out4


@Registers.model.register
class SpecificTrainVggNet19_bnBackboneLongModel(BaseModel):
    def __init__(self):
        super(SpecificTrainVggNet19_bnBackboneLongModel, self).__init__()
        self.extractor = Registers.module["VggNetBackbone"]('19_bn')
        self.fc = nn.Linear(512 * 4 * 4, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.extractor(input_tensor)
        long_out = output.view(batch_size, -1)
        long_out = self.fc(long_out)
        long_out2 = func.relu(long_out)
        long_out2 = self.dropout1(long_out2)
        long_out2 = self.fc2(long_out2)
        long_out3 = func.relu(long_out2)
        long_out3 = self.dropout2(long_out3)
        long_out3 = self.fc3(long_out3)
        long_out4 = func.relu(long_out3)
        long_out4 = self.dropout3(long_out4)
        long_out4 = self.fc4(long_out4)
        return long_out4


if __name__ == "__main__":
    import torchinfo

    model = SpecificTrainVggNet19BackboneLongModel(input_shape=())
    model.cuda()
    torchinfo.summary(model, ((4, 3, 128, 782), (4, 3, 128, 782), (4, 3, 128, 782)))
    # model = SpecificTrainResNet34BackboneLongModel(input_shape=())
    # # model.cuda()
    # torchinfo.summary(model, (4, 1, 128, 782))
