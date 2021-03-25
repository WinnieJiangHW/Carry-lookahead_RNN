import torch as t
import torch.nn.functional as F
import sys
sys.path.append("../")
from tcn import TemporalConvNet


class CRN(t.nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(CRN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = t.nn.Linear(784, output_size)

    def forward(self, inputs):
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear((y1 + inputs).view(-1, 784))
        return F.log_softmax(o, dim=1)
