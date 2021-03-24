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
        self.conv1 = t.nn.utils.weight_norm(t.nn.Conv1d(1, 1, 5, stride=5))
        self.relu1 = t.nn.ReLU()
        self.dropout1 = t.nn.Dropout(dropout)
        self.conv2 = t.nn.utils.weight_norm(t.nn.Conv1d(1, 1, 7, stride=7, padding=2))
        self.relu2 = t.nn.ReLU()
        self.dropout2 = t.nn.Dropout(dropout)
        self.net1 = t.nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.net2 = t.nn.Sequential(self.conv2, self.relu2, self.dropout2)
        self.linear = t.nn.Linear(23, output_size)

    def forward(self, x):
        c_in = t.zeros(x.shape[0], 1, 1).cuda()
        status = self.tcn(t.cat((c_in, x), dim=2))
        output_status = status[:, :, -1]
        status[:, :, 0] = t.zeros(x.shape[0], 1).cuda()
        status = status[:, :, 0:784].permute(0, 2, 1).reshape(-1, 1)
        x = x.view(-1, 1)
        result = self.cell(x, status).view(output_status.shape[0], 1, 784)
        result = self.net1(t.cat((result, c_in), dim=2))
        result = self.net2(result).view(-1, 23)
        return F.log_softmax(self.linear(result), dim=1)
