from torch import nn
import sys
sys.path.append("../")
from tcn import TemporalConvNet


class TCN_VRN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(TCN_VRN, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.cell = nn.RNNCell(input_size=input_size, hidden_size=input_size)
        self.decoder = nn.Linear(input_size, output_size)
        self.decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        emb = self.drop(self.encoder(x))
        y = self.tcn(emb.transpose(1, 2))
        z = self.cell(emb.view(-1, emb.shape[2]), y.transpose(1, 2).reshape(-1, y.shape[1]))
        o = self.decoder(z)
        return o.view(x.shape[0], x.shape[1], -1)
