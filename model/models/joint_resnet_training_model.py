from typing import Tuple

from torch import nn
import torch
import torch.nn.functional as func

from model.base_model import BaseModel
from model.manager import Register, Registers


@Registers.model.register
class SpecificTrainResNetModel(BaseModel):
    def __init__(self, input_shape: Tuple):
        super(SpecificTrainResNetModel, self).__init__()
        self.extractor = Registers.module["ResNet"](50)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        output = self.extractor(input_tensor)
        output = self.avg_pool(output)
        output = output.view(batch_size, -1)
        output = self.fc(output)
        return output


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

    model = ConcatModel()
    model.cuda()
    torchinfo.summary(model, (4, 6144, 8, 10))
