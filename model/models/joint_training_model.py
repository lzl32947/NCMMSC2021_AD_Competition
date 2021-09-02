from typing import Tuple

from torch import nn
import torch
import torch.nn.functional as func

from model.base_model import BaseModel
from model.manager import Register, Registers


class DenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(6240, 64)
        self.dropout = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(64, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = input_tensor.view((batch_size, -1))

        output = self.linear_1(output)
        output = func.relu(output, inplace=True)
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
class SpecificTrainModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super().__init__()
        self.extractor = ExtractionModel()
        self.dense = DenseModel()
        self.set_expected_input(input_shape)
        self.set_description("Specific 2D Train Model")

    def forward(self, input_tensor: torch.Tensor):
        output = self.extractor(input_tensor)
        output = self.dense(output)
        return output


@Registers.model.register
class MSMJointConcatFineTuneModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super().__init__()
        self.extractor_mfcc = ExtractionModel()
        self.extractor_spec = ExtractionModel()
        self.extractor_mel = ExtractionModel()
        self.dense = ConcatModel()
        self.set_expected_input(input_shape)
        self.set_description("MFCC SPEC MELSPEC Joint 2D Fine-tune Model")

    def forward(self, input_mfcc: torch.Tensor, input_spec: torch.Tensor, input_mel: torch.Tensor):
        output_mfcc = self.extractor_mfcc(input_mfcc)
        output_spec = self.extractor_spec(input_spec)
        output_mel = self.extractor_mel(input_mel)
        concat_output = torch.cat([output_spec, output_mel, output_mfcc], dim=1)
        output = self.dense(concat_output)
        return output
