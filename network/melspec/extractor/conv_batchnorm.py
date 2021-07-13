import torch
from torch import nn
import torch.nn.functional as func
import torchinfo
from torch.autograd import Variable
import os


class ExtractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(1, 16, (3, 3))
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv_layer_2 = nn.Conv2d(16, 32, (3, 3))
        self.batch_norm2 = nn.BatchNorm2d(32)

        self.conv_layer_3 = nn.Conv2d(32, 64, (3, 3))
        self.conv_layer_4 = nn.Conv2d(64, 128, (3, 3))

        self.conv_layer_5 = nn.Conv2d(128, 256, (3, 3))
        self.conv_layer_6 = nn.Conv2d(256, 512, (5, 5))

        self.bilstm_layer_1 = nn.LSTM(input_size=512, hidden_size=300, num_layers=2, bidirectional=True, batch_first=True)


        self._normal_init(self.conv_layer_1, 0, 0.01)
        self._normal_init(self.conv_layer_2, 0, 0.01)
        self._normal_init(self.conv_layer_3, 0, 0.01)
        self._normal_init(self.conv_layer_4, 0, 0.01)
        self._normal_init(self.conv_layer_5, 0, 0.01)
        self._normal_init(self.conv_layer_6, 0, 0.01)
        # self._lstm_init(self.bilstm_layer_1, 512, 13, 30)
        # self._lstm_init(self.bilstm_layer_2, 512, 13, 20)
        for name, param in self.bilstm_layer_1.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)




    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        output = self.conv_layer_1(input_tensor)
        output = func.relu(output, inplace=True)
        output = self.batch_norm1(output)

        output = self.conv_layer_2(output)
        output = func.relu(output, inplace=True)
        output = self.batch_norm2(output)

        output = nn.MaxPool2d(kernel_size=(4, 2), stride=(2, 1))(output)

        output = self.conv_layer_3(output)
        output = func.relu(output, inplace=True)

        output = nn.MaxPool2d(kernel_size=(4, 2), stride=(2, 2))(output)

        output = self.conv_layer_4(output)
        output = func.relu(output, inplace=True)

        output = nn.MaxPool2d(kernel_size=(4, 2), stride=(2, 2))(output)

        output = self.conv_layer_5(output)
        output = func.relu(output, inplace=True)

        output = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(output)

        output = self.conv_layer_6(output)
        output = func.relu(output, inplace=True)
        print(output.size())
        output = output.permute((0, 2, 3, 1))

        # h0 = torch.randn(2, 300)  # [bidirection*num_layers,batch_size,hidden_size]
        # c0 = torch.randn(2, 300)  # [bidirection*num_layers,batch_size,hidden_size]
        # output, (hn, cn) = self.bilstm_layer_1(output, (h0, c0))
        output = output.view(batch_size, 13, 512)
        print(output.size())
        hidden = Variable(torch.zeros(2 * 2, batch_size, 300))
        hidden =hidden.cuda()
        cell = Variable(torch.zeros(2 * 2, batch_size, 300))
        cell =cell.cuda()
        output ,(hidden_n, cell_n)= self.bilstm_layer_1(output, (hidden, cell))



        # output = output.permute((0, 1, 3, 2))
        # output = output.squeeze(dim=3)
        return output

    @staticmethod
    def _normal_init(m, mean, stddev, truncated=False):
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

    @staticmethod
    def _lstm_init(m, Embedding, s_len,hidden_size):

        input = torch.randn(s_len,Embedding)
        h0 = torch.randn(2, hidden_size)  # [bidirection*num_layers,batch_size,hidden_size]
        c0 = torch.randn(2, hidden_size)  # [bidirection*num_layers,batch_size,hidden_size]
        return input, h0, c0



if __name__=="__main__":
    torchinfo.summary(ExtractionModel().cuda(), (4, 1, 128, 157))
