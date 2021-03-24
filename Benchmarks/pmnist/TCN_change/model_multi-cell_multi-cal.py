import torch as t
import torch.nn.functional as F
import sys
sys.path.append("../../")
from TCN.tcn import TemporalConvNet
from multiprocessing.dummy import Pool


class CRN(t.nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(CRN, self).__init__()
        self.result = None
        self.status = None
        self.x = None
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout).cuda()
        self.cell = [t.nn.RNNCell(input_size=25, hidden_size=1).cuda()] * 784
        self.linear = t.nn.Linear(784, output_size).cuda()

    def rnncell(self, i):
        self.result[:, i] = self.cell[i](self.status[:, :, i].view([self.status.shape[0], self.status.shape[1]]), self.x[:, 0, i].view([self.x.shape[0], 1])).view(self.x.shape[0])

    def forward(self, x):
        self.x = x
        c_in = t.zeros(x.shape[0], 1, 1).cuda()
        self.status = self.tcn(t.cat((c_in, x[:, :, 0:783]), dim=2))
        c_in = t.zeros(x.shape[0], 25).cuda()
        self.result = t.empty(x.shape[0], 784).cuda()
        self.result[:, 0] = self.cell[0](c_in, x[:, 0, 0].view([x.shape[0], 1])).view(x.shape[0])
        i = list(range(1, 784))
        pool = Pool()
        pool.map(self.rnncell, i)
        pool.close()
        pool.join()
        return F.log_softmax(self.linear(self.result), dim=1).cuda()
