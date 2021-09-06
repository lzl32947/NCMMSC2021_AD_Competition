import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as func
from torch.nn import Parameter


def gelu(x):
    out = 1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    return out * x / 2


class FFL(nn.Module):
    def __init__(self, nx, nf):
        super(FFL, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size()[-1]), self.weight)
        return x.view(*size_out)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_fc = FFL(embed_dim, embed_dim * 4)
        self.conv_proj = FFL(embed_dim * 4, embed_dim)
        self.act = gelu
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, input_tensor):
        attention_output, _ = self.attention(input_tensor, input_tensor, input_tensor)
        norm = self.norm(attention_output + input_tensor)
        ffl = self.act(self.conv_fc(norm))
        out = self.conv_proj(ffl)
        norm_out = self.norm2(out + norm)
        return norm_out


class AttentionModule(nn.Module):
    def __init__(self, input_shape, block_num, num_heads):
        super().__init__()
        self.position_embedding = PositionalEncoding(input_shape[2], input_shape[1], 0.5)
        attention_blocks = AttentionBlock(input_shape[1], num_heads)
        self.out = nn.ModuleList([copy.deepcopy(attention_blocks) for i in range(block_num)])
        self.dense_1 = nn.Linear(input_shape[1] * input_shape[2], 1024)
        self.dropout_1 = nn.Dropout(0.5, inplace=True)
        self.dense_2 = nn.Linear(1024, 64)
        self.dropout_2 = nn.Dropout(0.5, inplace=True)
        self.dense_3 = nn.Linear(64, 3)

    def forward(self, input_tensor: torch.Tensor):
        batch = input_tensor.shape[0]
        out = self.position_embedding(input_tensor)
        out = torch.permute(out, [2, 0, 1])
        for block in self.out:
            out = block(out)
        out = out.permute([1, 0, 2]).contiguous()
        out = out.view(batch, -1)
        out = self.dense_1(out)
        out = self.dropout_1(out)
        out = self.dense_2(out)
        out = self.dropout_2(out)
        out = self.dense_3(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # 偶数sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # 奇数cos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).cuda()
        out = self.dropout(out)
        return out


if __name__ == '__main__':
    import torchinfo

    torchinfo.summary(AttentionModule((4, 20, 157), 20, 4).cuda(), (4, 20, 157))
