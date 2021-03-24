from torch import nn
import torch.nn.functional as F
import torch as t


class CRN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(CRN, self).__init__()
        self.hidden_size = num_channels[-1]
        self.lstm = nn.LSTM(input_size, num_channels[-1])
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        h0 = t.randn(1, x.shape[0], self.hidden_size).cuda()
        c0 = t.randn(1, x.shape[0], self.hidden_size).cuda()
        output, (hn, cn) = self.lstm(x.transpose(0, 1), (h0, c0))
        output = output.transpose(0, 1)
        output = self.linear(output).double()
        return self.sig(output)
