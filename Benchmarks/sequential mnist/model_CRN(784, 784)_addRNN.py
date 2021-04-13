import torch as t
import torch.nn.functional as F
import sys
sys.path.append("../")
from tcn import TemporalConvNet


class TCN_VRN(t.nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_VRN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.cell = t.nn.RNNCell(input_size=784, hidden_size=784)
        self.RNN = t.nn.RNN(1, 1)
        self.linear = t.nn.Linear(784, output_size)

    def forward(self, inputs):
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        y2 = self.cell(inputs.view(-1, 784), y1.view(-1, 784)).view(-1, 1, 784).permute(2, 0, 1)
        h0 = t.randn(1, inputs.shape[0], 1).cuda()
        y3, hn = self.RNN(y2, h0)
        y3 = y3.permute(1, 2, 0).view(-1, 784)
        o = self.linear(y3)
        return F.log_softmax(o, dim=1)
