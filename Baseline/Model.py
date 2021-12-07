from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, num_stock, d_market, d_alter, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2, t_mix = 1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=d_market + d_alter, hidden_size=hidn_rnn, num_layers=1, batch_first=False, dropout=dropout)
        self.fc = nn.Linear(2 * hidn_rnn, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_market, x_alter, relation_static=None):
        ## concat vs tensor
        x = torch.cat([x_market, x_alter], dim=-1)
        self.lstm.zero_grad()
        output, hidden = self.lstm(x)

        scores = self.sigmoid(self.fc(torch.cat(hidden, dim=-1)))

        return scores.squeeze()