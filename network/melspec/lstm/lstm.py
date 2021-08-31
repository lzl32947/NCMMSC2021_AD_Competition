import torch
from torch import nn
import torch.nn.functional as func


from network.melspec.lstm.lstm_layer import ExtractionModel
from network.melspec.lstm.lstm_layer import DenseModel
from network.melspec.lstm.lstm_layer import LstmModel


class MelSpecModel_lstm(nn.Module):
    def __init__(self):
        super().__init__()

        self.extractor = ExtractionModel()
        self.lstm = LstmModel()
        self.dense = DenseModel()

    def forward(self, input_tensor: torch.Tensor):
        output = self.extractor(input_tensor)
        output = self.lstm(output)
        output = self.dense(output)
        return output