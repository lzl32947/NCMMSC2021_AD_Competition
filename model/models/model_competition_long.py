from typing import Tuple

import torchvision.models
from torch import nn
import torch
import torch.nn.functional as func
from torch.utils import model_zoo

from model.base_model import BaseModel
from model.manager import Register, Registers
from model.modules.vgg import VggNetBackbone


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
class CompetitionSpecificTrainLongModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor = ExtractionModel()
        self.dense = DenseModel()
        self.pool = nn.AdaptiveAvgPool2d((13, 15))

    def forward(self, input_tensor: torch.Tensor):
        output = self.extractor(input_tensor)
        output = self.pool(output)
        output = self.dense(output)
        return output


@Registers.model.register
class CompetitionMSMJointConcatFineTuneLongModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor_mfcc = ExtractionModel()
        self.extractor_spec = ExtractionModel()
        self.extractor_mel = ExtractionModel()
        self.pool1 = nn.AdaptiveAvgPool2d((13, 15))
        self.pool2 = nn.AdaptiveAvgPool2d((13, 15))
        self.pool3 = nn.AdaptiveAvgPool2d((13, 15))
        self.dense = ConcatModel()

    def forward(self, input_mfcc: torch.Tensor, input_spec: torch.Tensor, input_mel: torch.Tensor):
        output_mfcc = self.extractor_mfcc(input_mfcc)
        output_spec = self.extractor_spec(input_spec)
        output_mel = self.extractor_mel(input_mel)
        output_mfcc = self.pool1(output_mfcc)
        output_spec = self.pool2(output_spec)
        output_mel = self.pool3(output_mel)
        concat_output = torch.cat([output_spec, output_mel, output_mfcc], dim=1)
        output = self.dense(concat_output)
        return output


@Registers.model.register
class CompetitionMSMJointConcatFineTuneResNet18BackboneLongModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.extractor_mfcc = Registers.module["ResNetBackbone"](18)
        self.extractor_spec = Registers.module["ResNetBackbone"](18)
        self.extractor_mel = Registers.module["ResNetBackbone"](18)
        self.dense = ResNet18ConcatModel()

    def forward(self, input_mfcc: torch.Tensor, input_spec: torch.Tensor, input_mel: torch.Tensor):
        output_mfcc = self.extractor_mfcc(input_mfcc)
        output_spec = self.extractor_spec(input_spec)
        output_mel = self.extractor_mel(input_mel)
        concat_output = torch.cat([output_spec, output_mel, output_mfcc], dim=1)
        output = self.dense(concat_output)
        return output


class ResNet18ConcatModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(512 * 3, 512, (3, 3))

        self.maxpooling_2 = nn.MaxPool2d((2, 2))

        self.linear_1 = nn.Linear(512 * 11, 512)
        self.dropout_1 = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(512, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output)

        output = self.maxpooling_2(output)

        output = output.view((batch_size, -1))
        output = self.linear_1(output)
        output = self.dropout_1(output)
        output = self.linear_2(output)
        return output


@Registers.model.register
class SpecificTrainVggNet19BackboneLongModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super(SpecificTrainVggNet19BackboneLongModel, self).__init__()
        self.extractor = Registers.module["VggNetBackbone"](19)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.extractor(input_tensor)
        long_out = output.view(batch_size, -1)
        long_out = self.fc(long_out)
        long_out2 = func.relu(long_out)
        long_out2 = self.fc2(long_out2)
        long_out3 = func.relu(long_out2)
        long_out3 = self.fc3(long_out3)
        return long_out3


@Registers.model.register
class CompetitionSpecificTrainVggNet19BNBackboneLongModel(BaseModel):
    def __init__(self):
        super(CompetitionSpecificTrainVggNet19BNBackboneLongModel, self).__init__()
        # self.extractor = Registers.module["VggNetBackbone"]("19_bn")
        self.extractor = VggNetBackbone("19_bn")
        self.conv1 = nn.Conv2d(512, 1024, (3, 3))
        self.max_pooling = nn.MaxPool2d((2, 2), stride=2)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.extractor(input_tensor)
        output = self.conv1(output)
        output = func.relu(output)
        output = self.max_pooling(output)
        output = self.pooling(output)
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
class CompetitionMSMJointTrainVggNet19BNBackboneLongModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super().__init__()
        self.extractor_mfcc = Registers.module["VggNetBackbone"]("19_bn")
        self.extractor_spec = Registers.module["VggNetBackbone"]("19_bn")
        self.extractor_mel = Registers.module["VggNetBackbone"]("19_bn")
        self.conv1 = nn.Conv2d(512, 1024, (3, 3))
        self.max_pooling = nn.MaxPool2d((2, 2), stride=2)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = VggNet19BNLongConcatModel()
        self.set_expected_input(input_shape)
        self.set_description("MFCC SPEC MELSPEC Joint 2D Fine-tune Model")

    def forward(self, input_mfcc: torch.Tensor, input_spec: torch.Tensor, input_mel: torch.Tensor):
        output_mfcc = self.extractor_mfcc(input_mfcc)
        output_mfcc = self.conv1(output_mfcc)
        output_mfcc = self.max_pooling(output_mfcc)
        output_mfcc = self.pooling(output_mfcc)

        output_spec = self.extractor_spec(input_spec)
        output_spec = self.conv1(output_spec)
        output_spec = self.max_pooling(output_spec)
        output_spec = self.pooling(output_spec)

        output_mel = self.extractor_mel(input_mel)
        output_mel = self.conv1(output_mel)
        output_mel = self.max_pooling(output_mel)
        output_mel = self.pooling(output_mel)
        concat_output = torch.cat([output_spec, output_mel, output_mfcc], dim=1)
        output = self.dense(concat_output)
        return output


class VggNet19BNLongConcatModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(512 * 3, 512, (3, 3))

        self.maxpooling_2 = nn.MaxPool2d((2, 2))

        self.linear_1 = nn.Linear(512 * 11, 512)
        self.dropout_1 = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(512, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output)

        output = self.maxpooling_2(output)

        output = output.view((batch_size, -1))
        output = self.linear_1(output)
        output = self.dropout_1(output)
        output = self.linear_2(output)
        return output


if __name__ == "__main__":
    import torchinfo

    model = CompetitionSpecificTrainVggNet19BNBackboneLongModel()
    model.cuda()
    torchinfo.summary(model, (4, 3, 128, 782))
    # model = SpecificTrainResNet34BackboneLongModel(input_shape=())
    # # model.cuda()
    # torchinfo.summary(model, (4, 1, 128, 782))
