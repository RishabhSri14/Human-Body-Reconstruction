# Write a program for a positional encoder for a nerf algorithm.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoder(nn.Module):
    
    def __init__(self, d_model, max_seq_len=100):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
                self.pe[pos, i+1] = np.cos(pos / (10000 ** ((2 * (i+1))/d_model)))
        self.pe = self.pe.unsqueeze(0)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# Define encoder
encoder = PositionalEncoder(512)

# An example of positional encoder which take a torch tensor as input
# and return a torch tensor as output.

# Example:
# encoder = PositionalEncoder(512)
x = torch.randn(1, 100, 512)
x = encoder(x)
print(x.shape)

# Path: decoder.py
# Write a program for a positional decoder for a nerf algorithm.
