from torch import nn
import sys
sys.path.append("../")
from tcn import TemporalConvNet
import torch.nn.functional as F


class CL_RNN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(CL_RNN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.cell = nn.RNNCell(input_size, num_channels[-1])
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.cell(x.view(-1, x.shape[2]), output.view(-1, output.shape[2])).view(1, output.shape[1], -1)
        output = self.linear(output).double()
        return self.sig(output)
