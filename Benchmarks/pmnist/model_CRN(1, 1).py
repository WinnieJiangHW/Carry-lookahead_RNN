import torch as t
import torch.nn.functional as F
import sys
sys.path.append("../../")
from TCN.tcn import TemporalConvNet


class CRN(t.nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(CRN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.cell = t.nn.RNNCell(input_size=1, hidden_size=1)
        self.linear = t.nn.Linear(784, output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        y2 = self.cell(inputs.view(-1, 1), y1.view(-1, 1))
        o = self.linear(y2.view(-1, 784))
        return F.log_softmax(o, dim=1)
