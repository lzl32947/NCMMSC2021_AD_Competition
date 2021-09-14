from typing import Tuple

import torchvision.models
from torch import nn
import torch
import torch.nn.functional as func
from torch.utils import model_zoo

from model.base_model import BaseModel
from model.manager import Register, Registers

@Registers.model.register
class SpecificTrainResNet18BackboneLongModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super(SpecificTrainResNet18BackboneLongModel, self).__init__()
        self.extractor = Registers.module["ResNetBackbone"](18)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 6))
        self.fc = nn.Linear(512 * 6, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 3)

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


@Registers.model.register
class SpecificTrainVggNet19BackboneLongModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super(SpecificTrainVggNet19BackboneLongModel, self).__init__()
        self.extractor = Registers.module["VggNetBackbone"](19)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.fc = nn.Linear(512*7*7, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        # print(batch_size)
        output = self.extractor(input_tensor)
        # print(output.shape)
        # output = self.avg_pool(output)
        long_out = output.view(batch_size, -1)
        long_out = self.fc(long_out)
        long_out2 = func.relu(long_out)
        long_out2 = self.fc2(long_out2)
        long_out3 = func.relu(long_out2)
        long_out3 = self.fc3(long_out3)
        return long_out3


@Registers.model.register
class MSMJointFusionFineTuneModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super().__init__()
        self.extractor_mfcc = ExtractionModel()
        self.extractor_spec = ExtractionModel()
        self.extractor_mel = ExtractionModel()
        self.fusion_mfcc_spec = Registers.module["IAFFFusion"]((13, 15), 32, 4)
        self.fusion_mfcc_melspec = Registers.module["IAFFFusion"]((13, 15), 32, 4)
        self.fusion_specs = Registers.module["IAFFFusion"]((13, 15), 32, 4)
        self.dense = FusionDenseModel()
        self.set_expected_input(input_shape)
        self.set_description("MFCC SPEC MELSPEC Joint 2D Fusion Fine-tune Model")

    def forward(self, input_mfcc: torch.Tensor, input_spec: torch.Tensor, input_mel: torch.Tensor):
        output_mfcc = self.extractor_mfcc(input_mfcc)
        output_spec = self.extractor_spec(input_spec)
        output_mel = self.extractor_mel(input_mel)
        fusion_mfcc_spec = self.fusion_mfcc_spec(output_mfcc, output_spec)
        fusion_mfcc_mel = self.fusion_mfcc_spec(output_mfcc, output_mel)
        fusion_specs = self.fusion_mfcc_spec(fusion_mfcc_spec, fusion_mfcc_mel)
        output = self.dense(fusion_specs)
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

class FusionDenseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(32, 256, (3, 3))

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