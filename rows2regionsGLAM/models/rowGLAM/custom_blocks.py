import torch
from torch.nn import ModuleList, Linear, ReLU, GELU, Softmax, Sigmoid
from torch_geometric.nn import BatchNorm, TAGConv, GCNConv, GATConv

activations = {
    "gelu": GELU(),
    "relu": ReLU(),
    "softmax": Softmax(),
    "sigmoid": Sigmoid(),
    "none": None
}

gnn_convs = {
    "tag": lambda params: TAGConv(params['in_gnn'], params['out_gnn'], K=params['K']),
    "conv": lambda params: GCNConv(params['in_gnn'], params['out_gnn']),
    "gat": lambda params: GATConv(params['in_gnn'], params['out_gnn']),
}

class GNNBlock(torch.nn.Module):
    def __init__(self, type_conv, params):
        super(GNNBlock, self).__init__()
        self.is_exist_batch_norm = "batch_norm" in params and params['batch_norm'] 
        self.is_concat = "concat" in params and params['concat']

        self.linear = Linear(params['in'],params['in_gnn'])
        if self.is_exist_batch_norm:
            self.batch_norm = BatchNorm(params['in_gnn'])
        self.gnn_activation = activations[params['gnn_activation']]
        self.activation = activations[params['activation']]
        self.gnn_conv = gnn_convs[type_conv](params)
        

    def forward(self, x, edge_index):
        x_in = x
        x = self.linear(x)
        if self.is_exist_batch_norm:
            x = self.batch_norm(x)
        if self.gnn_activation is not None:
            x = self.gnn_activation(x)
        x = self.gnn_conv(x, edge_index)
        if self.is_concat:
            x = torch.cat((x, x_in), dim=-1)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Block(torch.nn.Module):
    def __init__(self, params):
        super(Block, self).__init__()
        self.is_exist_linear_pred = "linear_pred" in params and len(params['linear_pred']) != 0
        self.is_exist_linear_post = "linear_post" in params and len(params['linear_post']) != 0
        self.is_exist_gnn = "gnn" in params and len(params['gnn']) != 0

        if self.is_exist_linear_pred:
            self.linear_pred = ModuleList([
                Linear(linear['in'], linear['out']) for linear in params['linear_pred']
            ])
            self.activation_pred = [
                activations[linear['activation']] for linear in params['linear_pred']
            ]
        if self.is_exist_linear_post:
            self.linear_post = ModuleList([
                Linear(linear['in'], linear['out']) for linear in params['linear_post']
            ])
            self.activation_post = [
                activations[linear['activation']] for linear in params['linear_post']
            ]
        if self.is_exist_gnn:
            self.gnns = ModuleList([
                GNNBlock(type_conv, params_conv) for type_conv, params_conv in params['gnn']
            ]) 
        

    def forward(self, x, edge_index):
        if self.is_exist_linear_pred:
            for linear, activation in zip(self.linear_pred, self.activation_pred):
                x = linear(x)
                if activation is not None:
                    x = activation(x)
        if self.is_exist_gnn:
            for gnn in self.gnns:
                x = gnn(x, edge_index)
        if self.is_exist_linear_post:
            for linear, activation in zip(self.linear_post, self.activation_post):
                x = linear(x)
                if activation is not None:
                    x = activation(x)
        return x