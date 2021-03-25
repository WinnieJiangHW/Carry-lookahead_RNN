import torch as t
import torch.nn.functional as F


class LSTM(t.nn.Module):
    def __init__(self, output_size):
        super(LSTM, self).__init__()
        self.LSTM = t.nn.LSTM(1, 1)
        self.linear = t.nn.Linear(784, output_size)

    def forward(self, inputs):
        h0 = t.randn(1, inputs.shape[0], 1).cuda()
        c0 = t.randn(1, inputs.shape[0], 1).cuda()
        y3, (hn, cn) = self.LSTM(inputs.permute(2, 0, 1), (h0, c0))
        y3 = y3.permute(1, 2, 0).view(-1, 784)
        o = self.linear(y3)
        return F.log_softmax(o, dim=1)
