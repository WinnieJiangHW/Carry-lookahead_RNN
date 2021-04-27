import torch as t
import torch.nn.functional as F
import sys
sys.path.append("../")
from tcn import TemporalConvNet


class CL_RNN(t.nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(CL_RNN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.cell = t.nn.RNNCell(input_size=784, hidden_size=784)
        self.relu = t.nn.ReLU()
        self.dropout = t.nn.Dropout(0.05)
        self.conv1 = t.nn.utils.weight_norm(t.nn.Conv1d(1, 1, 5, stride=5))
        self.relu1 = t.nn.ReLU()

        self.dropout1 = t.nn.Dropout(dropout)
        self.conv2 = t.nn.utils.weight_norm(t.nn.Conv1d(1, 1, 7, stride=7, padding=2))
        self.relu2 = t.nn.ReLU()
        self.dropout2 = t.nn.Dropout(dropout)
        self.net1 = t.nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.net2 = t.nn.Sequential(self.conv2, self.relu2, self.dropout2)
        self.linear = t.nn.Linear(23, output_size)

    def forward(self, inputs):
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        y2 = self.cell(inputs.view(-1, 784), y1.view(-1, 784))
        y3 = self.dropout(self.relu(y2))
        padding = t.zeros(inputs.shape[0], 1, 1).cuda()
        y4 = self.net1(t.cat((y3.view(inputs.shape[0], 1, 784), padding), dim=2))
        o = self.net2(y4).view(-1, 23)
        return F.log_softmax(self.linear(o), dim=1)
