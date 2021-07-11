import torch
from torch import nn
import torch.nn.functional as func


class DenseModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 2048)
        self.linear2 = nn.Linear(2048, 16)
        self.linear3 = nn.Linear(16, 3)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = input_tensor.view((batch_size, -1))
        output = self.linear1(output)
        output = func.relu(output, inplace=True)

        output = self.linear2(output)
        output = func.relu(output, inplace=True)

        output = self.linear3(output)
        output = self.softmax(output)
        return output
