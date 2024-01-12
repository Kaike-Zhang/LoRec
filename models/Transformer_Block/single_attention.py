import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        # 1) compute attention score (query x key then / dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        # 1.5) if the attention type is mask-attention then change the masking part to 0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 2) do softmax
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        # 3) the output is product of p_attn and value
        return torch.matmul(p_attn, value), p_attn
