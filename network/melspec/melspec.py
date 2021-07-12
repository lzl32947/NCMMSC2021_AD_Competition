import torch
from torch import nn
import torch.nn.functional as func

from network.melspec.extractor.standard_conv import ExtractionModel


class MelSpecModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.extractor = ExtractionModel()

    def forward(self, input_tensor: torch.Tensor):
        output = self.extractor(input_tensor)
        return output
