import torch
from torch import nn
import torch.nn.functional as func

from network.melspec.extractor.conv_batchnorm import ExtractionModel
from network.melspec.stand_alone.denses_conv_batchnorm import DenseModel


class MelSpecModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.extractor = ExtractionModel()
        self.dense = DenseModel(6656)

    def forward(self, input_tensor: torch.Tensor):
        output = self.extractor(input_tensor)
        output = self.dense(output)
        return output
