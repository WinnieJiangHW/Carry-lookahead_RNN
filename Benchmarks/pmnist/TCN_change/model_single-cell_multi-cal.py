import torch as t
import torch.nn.functional as F
import sys
sys.path.append("../../")
from TCN.tcn import TemporalConvNet


class CRN(t.nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(CRN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout).cuda()
        self.cell = t.nn.RNNCell(input_size=25, hidden_size=1).cuda()
        self.linear = t.nn.Linear(784, output_size).cuda()

    def forward(self, x):
        c_in = t.zeros(x.shape[0], 1, 1).cuda()
        status = self.tcn(t.cat((c_in, x), dim=2))
        output_status = status[:, :, -1]
        status[:, :, 0] = t.zeros(x.shape[0], 25).cuda()
        status = status[:, :, 0:784]
        result = t.empty(x.shape[0], 784).cuda()
        for i in range(784):
            result[:, i] = self.cell(status[:, :, i].view(status.shape[0], 25), x[:, 0, i].view(x.shape[0], 1)).view(x.shape[0])
        return F.log_softmax(self.linear(result), dim=1).cuda()
