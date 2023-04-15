# Write a program for a positional encoder for a nerf algorithm.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoder(nn.Module):
    
    def __init__(self, d_model, num_freq=10):
        super(PositionalEncoder, self).__init__()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.d_model = d_model
        self.max_seq_len = num_freq
        # self.pe = torch.zeros(d_model, max_seq_len,2)
        self.sinus_in=torch.arange(0,self.max_seq_len,dtype=torch.int8).to(self.device)
        self.sinus_in=self.sinus_in[None,None,:]
        print(self.sinus_in.shape)
        # for pos in range(max_seq_len):
        #     for i in range(0, d_model, 2):
        #         self.pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
        #         self.pe[pos, i+1] = np.cos(pos / (10000 ** ((2 * (i+1))/d_model)))
        # self.pe = self.pe.unsqueeze(0)
        
    def forward(self, x):
        out=torch.zeros((x.shape[0],x.shape[1],self.max_seq_len,2)).to(self.device)
        out[..., 0] = torch.sin(2*x.unsqueeze(-1)*self.sinus_in).to(self.device)
        out[..., 1] = torch.cos(2*x.unsqueeze(-1)*self.sinus_in).to(self.device)
        # x = x + self.pe[:, :x.size(1)]

        return out.reshape(out.shape[:-2]+(-1,))

if __name__ == '__main__':
    # Define encoder
    encoder = PositionalEncoder(5,10)

    # An example of positional encoder which take a torch tensor as input
    # and return a torch tensor as output.

    # Example:
    # encoder = PositionalEncoder(512)
    # x = torch.randn(1, 100, 512)
    x=torch.randn(100,5)
    print(x.shape)
    x = encoder(x)
    print(x.shape)
