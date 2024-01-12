import torch
import torch.nn as nn

import math


class PositionEmb(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionEmb, self).__init__()
        pe = torch.zeros(max_len, d_model)
        po = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(po * div_term)
        pe[:, 1::2] = torch.cos(po * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        _, l, _ = x.shape
        return self.pe[:l, :].unsqueeze(0)



