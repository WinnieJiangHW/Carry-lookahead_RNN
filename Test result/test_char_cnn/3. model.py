from torch import nn
import torch as t


class CRN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(CRN, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.LSTM = nn.LSTM(input_size, input_size)
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
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        h0 = t.randn(1, emb.shape[0], emb.shape[2]).cuda()
        c0 = t.randn(1, emb.shape[0], emb.shape[2]).cuda()
        r, (hn, cn) = self.LSTM(emb.permute(1, 0, 2), (h0, c0))
        r = r.permute(1, 0, 2)
        o = self.decoder(r)
        return o.view(x.shape[0], x.shape[1], -1)