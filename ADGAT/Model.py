from .Layers import *


class AD_GAT(nn.Module):
    def __init__(self, num_stock, d_market, d_alter, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2, t_mix = 1):
        super(AD_GAT, self).__init__()
        self.t_mix = t_mix
        self.dropout = dropout
        if  self.t_mix == 0: # concat
            self.GRUs_s = Graph_GRUModel(num_stock, d_market + d_alter, hidn_rnn)
            self.GRUs_r = Graph_GRUModel(num_stock, d_market + d_alter, hidn_rnn)
        elif self.t_mix == 1: # all_tensor
             self.tensor = Graph_Tensor(num_stock,d_hidden,d_market,d_alter)
             self.GRUs_s = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
             self.GRUs_r = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
        self.attentions = [
            Graph_Attention(hidn_rnn, hidn_att, num_stock=num_stock, dropout=dropout, alpha=alpha, residual=True, concat=True) for _
            in range(heads_att)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.X2Os = Graph_Linear(heads_att * hidn_att  + hidn_rnn , 1, bias = True)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def get_gate(self,x_numerical,x_textual):
        x_s = self.tensor(x_numerical, x_textual)
        x_s = self.GRUs_s(x_s)
        gate = torch.stack([att.get_gate(x_s) for att in self.attentions])
        return gate

    def forward(self, x_market, x_alter, relation_static = None):
        ## concat vs tensor
        if self.t_mix == 0:  # concat
            x_s = torch.cat([x_market, x_alter], dim=-1)
            x_r = torch.cat([x_market, x_alter], dim=-1)
        elif self.t_mix == 1:  # concat
            x_s = self.tensor(x_market, x_alter)
            x_r = self.tensor(x_market, x_alter)
        #GRUs for extract different sequential embedding for relation/gate inferring.
        #Equivalent to use a single GRU and separate non-linear decoders.
        x_r = self.GRUs_r(x_r)
        x_s = self.GRUs_s(x_s)
        x_r = F.dropout(x_r, self.dropout, training=self.training)
        x_s = F.dropout(x_s, self.dropout, training=self.training)
        ##
        x = torch.cat([att(x_s, x_r, relation_static = relation_static) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([x, x_s], dim=1)
        x = self.X2Os(x)
        output = torch.sigmoid(x).squeeze(-1)
        return output


class DeepQuant(nn.Module):
    def __init__(self, num_stock, d_market, d_alter, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2, t_mix = 1):
        super(DeepQuant, self).__init__()
        self.ad_gat = AD_GAT(num_stock, d_market, d_alter, d_hidden, hidn_rnn, heads_att, hidn_att, dropout, alpha, t_mix)
        self.lstm = nn.LSTM(input_size=d_market + d_alter, hidden_size=hidn_rnn, num_layers=1, batch_first=False, dropout=dropout)
        self.fc1 = nn.Linear(2 * hidn_rnn, 1)
        self.fc2 = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_market, x_alter, relation_static = None):
        ## concat vs tensor
        self.ad_gat.zero_grad()
        self.lstm.zero_grad()
        graph_output = self.ad_gat(x_market, x_alter, relation_static).unsqueeze(-1)
        x = torch.cat([x_market, x_alter], dim=-1)
        output, hidden = self.lstm(x)
        highway = self.relu(self.fc1(torch.cat(hidden, dim=-1))).squeeze(0)
        scores = self.fc2(torch.cat([graph_output, highway], dim=-1))
        output = self.sigmoid(scores).squeeze(-1)

        return output