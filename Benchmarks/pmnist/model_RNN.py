import torch as t
import torch.nn.functional as F
import sys
sys.path.append("../../")


class RNN(t.nn.Module):
    def __init__(self, output_size):
        super(RNN, self).__init__()
        self.RNN = t.nn.RNN(1, 1)
        self.linear = t.nn.Linear(784, output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        h0 = t.randn(1, inputs.shape[0], 1).cuda()
        y3, hn = self.RNN(inputs.permute(2, 0, 1), h0)
        y3 = y3.permute(1, 2, 0).view(-1, 784)
        o = self.linear(y3)
        return F.log_softmax(o, dim=1)
