from .utils import *
import torch.nn.functional as F

class Graph_Linear(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(Graph_Linear, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(input_size,hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x):
        output = x @ self.W
        if self.bias:
            output = output + self.b
        return output


class Graph_Tensor(nn.Module):
    def __init__(self, num_stock, d_hidden, d_market, d_alternative, bias=True):
        super(Graph_Tensor, self).__init__()
        self.num_stock = num_stock
        self.d_hidden = d_hidden
        self.d_market = d_market
        self.d_alternative = d_alternative
        self.seq_transformation_alternative = nn.Conv1d(d_alternative, d_hidden, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_markets = nn.Conv1d(d_market, d_hidden, kernel_size=1, stride=1, bias=False)
        self.tensorGraph = nn.Parameter(torch.zeros(num_stock, d_hidden, d_hidden, d_hidden))
        self.W = nn.Parameter(torch.zeros(num_stock, 2 * d_hidden, d_hidden))
        self.b = nn.Parameter(torch.zeros(num_stock, d_hidden))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, market, alternative):
        t, num_stocks = alternative.size()[0], alternative.size()[1]

        alternative_transformed = alternative.reshape(-1, self.d_alternative)
        alternative_transformed = torch.transpose(alternative_transformed, 0, 1).unsqueeze(0)
        alternative_transformed = self.seq_transformation_alternative(alternative_transformed)
        alternative_transformed = alternative_transformed.squeeze().transpose(0, 1)
        alternative_transformed = alternative_transformed.reshape(t, num_stocks, self.d_hidden)

        market_transformed = market.reshape(-1, self.d_market)
        market_transformed = torch.transpose(market_transformed, 0, 1).unsqueeze(0)
        market_transformed = self.seq_transformation_markets(market_transformed)
        market_transformed = market_transformed.squeeze().transpose(0, 1)
        market_transformed = market_transformed.reshape(t, num_stocks, self.d_hidden)

        x_alternative_tensor = alternative_transformed.unsqueeze(2)
        x_alternative_tensor = x_alternative_tensor.unsqueeze(2)
        x_market_tensor = market_transformed.unsqueeze(-1)
        temp_tensor = x_alternative_tensor.matmul(self.tensorGraph).squeeze()
        temp_tensor = temp_tensor.matmul(x_market_tensor).squeeze()
        x_linear = torch.cat((alternative_transformed, market_transformed), axis=-1)
        temp_linear = torch.bmm(x_linear.transpose(0, 1), self.W)
        temp_linear = temp_linear.transpose(0, 1)

        output = torch.tanh(temp_tensor + temp_linear + self.b)
        return output

class Graph_GRUCell(nn.Module):
    def __init__(self, num_nodes, input_size, hidden_size, bias=True):
        super(Graph_GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = Graph_Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = Graph_Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x, hidden):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy

class Graph_GRUModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, bias=True):
        super(Graph_GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = Graph_GRUCell(num_nodes, input_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.size()[1], self.hidden_dim, device=x.device,dtype = x.dtype)
        for seq in range(x.size(0)):
            hidden = self.gru_cell(x[seq], hidden)
        return hidden

class Graph_Attention(nn.Module):

    def __init__(self, in_features, out_features, num_stock, dropout, alpha ,concat=True, residual=False):
        super(Graph_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.residual = residual

        self.seq_transformation_r = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_s = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)

        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)

        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)

        self.W_static = nn.Parameter(torch.zeros(num_stock, num_stock).type(torch.FloatTensor), requires_grad=True)

        self.w_1 = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
        self.w_2 = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)

        self.coef_revise = False
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def get_relation(self, input_r, relation_static = None):
        num_stock = input_r.shape[0]
        seq_r = torch.transpose(input_r, 0, 1).unsqueeze(0)
        logits = torch.zeros(num_stock, num_stock, device=input_r.device, dtype=input_r.dtype)
        seq_fts_r = self.seq_transformation_r(seq_r)
        f_1 = self.f_1(seq_fts_r)
        f_2 = self.f_2(seq_fts_r)
        logits += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
        if relation_static != None:
            logits += torch.mul(relation_static, self.W_static)
        coefs = F.elu(logits)
        if not isinstance(self.coef_revise,torch.Tensor):
            num_stock = relation_static.shape[0]
            self.coef_revise = torch.zeros(num_stock, num_stock, device = input_r.device) + 1.0 - torch.eye(num_stock, num_stock, device = input_r.device)
        coefs_eye = coefs.mul(self.coef_revise)
        return coefs_eye

    def get_gate(self, seq_s):
        transform_1 = self.w_1(seq_s)
        transform_2 = self.w_2(seq_s)
        transform_1 = torch.transpose(transform_1.squeeze(0), 0, 1)
        transform_2 = torch.transpose(transform_2.squeeze(0), 0, 1)
        gate = F.elu(transform_1.unsqueeze(1) + transform_2)
        return gate

    def forward(self, input_s, input_r, relation_static = None):
        # unmasked attention
        coefs_eye = self.get_relation(input_r, relation_static)
        # attribute-mattered propagation
        seq_s = torch.transpose(input_s, 0, 1).unsqueeze(0)
        seq_fts_s = self.seq_transformation_s(seq_s)
        seq_fts_s = F.dropout(torch.transpose(seq_fts_s.squeeze(0), 0, 1), self.dropout, training=self.training)
        #
        gate = self.get_gate(seq_s)
        #
        seq_fts_s_gated = seq_fts_s * gate
        ret = torch.bmm(coefs_eye.unsqueeze(1), seq_fts_s_gated).squeeze()
        if self.concat:
            return torch.tanh(ret)
        else:
            return ret

