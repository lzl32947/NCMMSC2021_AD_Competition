import torch
from torch import nn


from network.melspec.lstm.lstm_layer import ExtractionModel
from network.melspec.lstm.lstm_layer import LstmModel


class MelSpecModel_lstm(nn.Module):
    def __init__(self):
        super().__init__()

        self.extractor = ExtractionModel()
        self.lstm = LstmModel()

    def forward(self, input_tensor: torch.Tensor):
        output = self.extractor(input_tensor)
        output = self.lstm(output)
        return output