from typing import List
import numpy as np
import networkx as nx
import torch
from torch.nn import Linear, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, GELU, HuberLoss,ModuleList, ReLU
from torch.nn.functional import relu
from torch_geometric.nn import BatchNorm, TAGConv, GCNConv, GATConv
import random


class CustomLoss(torch.nn.Module):
    def __init__(self, params):
        super(CustomLoss, self).__init__()

        self.bce = BCEWithLogitsLoss(pos_weight=torch.tensor(params['edge_imbalance']))
        self.ce = CrossEntropyLoss(weight=torch.tensor(params['publaynet_imbalance']))
        self.edge_coef: float = params['edge_coef']
        self.node_coef: float = params['node_coef']

    def forward(self, pred_dict, dict_graph):
        e_pred = pred_dict["E_pred_logits"]
        e_true = dict_graph["true_edges"]
        e_pred = e_pred.squeeze()

        loss_edge = self.bce(e_pred, e_true)

        n_pred = pred_dict["node_classes"]

        n_true = dict_graph["true_nodes"]

        loss_node = self.ce(n_pred, n_true)

        loss = self.edge_coef * loss_edge + self.node_coef * loss_node
        return loss


class TorchModel(torch.nn.Module):

    def __init__(self, params):
        super(TorchModel, self).__init__()
        self.activations = {
            "gelu": GELU,
            "relu": ReLU,
            "softmax": torch.nn.Softmax,
            "sigmoid": torch.nn.Sigmoid,
        }
        self.params = params
        self.sigmoid = torch.nn.Sigmoid()
        self.make_model()

    def forward(self, data_graph_dict):
        # Подготовка данных
        X: torch.Tensor = data_graph_dict["X"]
        Y: torch.Tensor = data_graph_dict["Y"]
        sp_A: torch.Tensor = data_graph_dict["sp_A"]
        A = sp_A.to_dense()
        new_A = self.build_conjugate_graph(A)
        new_sp_A = new_A.to_sparse_coo()
        inds: List[int] = data_graph_dict["inds"]
        model_type = self.params["model_type"]
        x = X

        if not (self.mlp_pred is None):
            x = self.mlp_pred(x)
        x = self.forward_gnn(x, sp_A, self.gcn_node)
        match model_type:
            case 1:

                Node_classes = self.mlp_node_class(x)
                Omega = torch.cat([x[inds[0]], x[inds[1]], X[inds[0]], X[inds[1]], Y], dim=1)
                Edge_pred_logits = self.mlp_edge_class(Omega)
            case 2:
                Node_classes = self.mlp_node_class(x)
                if not (self.mlp_node_pred is None):
                    x = self.mlp_node_pred(x)
                x = self.forward_gnn(x, sp_A, self.gcn_node_post)
                x = self.mlp_node_edge(x)
                Omega = torch.cat([x[inds[0]], x[inds[1]], X[inds[0]], X[inds[1]], Y], dim=1)
                Edge_pred_logits = self.mlp_edge_class(Omega)
            case 3:

                Node_classes = self.mlp_node_class(x)
                Omega = torch.cat([x[inds[0]], x[inds[1]], X[inds[0]], X[inds[1]], Y], dim=1)
                Omega = self.mlp_edge_pred(Omega)
                Omega = self.forward_gnn(Omega, new_sp_A, self.gcn_edge)
                Edge_pred_logits = self.mlp_edge_class(Omega)

        return {
            "node_classes": Node_classes,
            "E_pred_logits": Edge_pred_logits,
            "E_pred": self.sigmoid(Edge_pred_logits)
        }

    def forward_gnn(self, x, adj, gnn):
        for modlist in gnn:
            h = x
            for layer in modlist:
                if isinstance(layer, (TAGConv, GATConv, GCNConv)):
                    x = layer(x, adj)
                else:
                    x = layer(x)

            if self.params['concat_gcn']:
                z = torch.concat([h, x], dim=-1)
            else:
                z = x
            x = z
        return x

    def make_model(self):
        model_type = self.params["model_type"]

        dim = self.params["node_featch"]
        self.mlp_pred, dim = (None, dim) if self.params["mlp_pred"] is None else self.build_mlp_layer(
            self.params["mlp_pred"], dim)
        match model_type:
            case 1:
                print(dim)
                self.gcn_node, dim = self.build_gcn_layer(self.params["gcn_node"], dim)
                self.mlp_node_class, _ = self.build_mlp_layer(self.params["mlp_node_class"], dim)
                dim = self.params["node_featch"] * 2 + self.params["edge_featch"] + dim * 2
                self.mlp_edge_class, _ = self.build_mlp_layer(self.params["mlp_edge_class"], dim)
            case 2:

                self.gcn_node, dim = self.build_gcn_layer(self.params["gcn_node"], dim)
                self.mlp_node_class, _ = self.build_mlp_layer(self.params["mlp_node_class"], dim)
                self.mlp_node_pred, dim = (None, dim) if self.params["mlp_node_pred"] is None else self.build_mlp_layer(
                    self.params["mlp_node_pred"], dim)
                self.gcn_node_post, dim = self.build_gcn_layer(self.params["gcn_node_post"], dim)
                self.mlp_node_edge, dim = self.build_mlp_layer(self.params["mlp_node_edge"], dim)
                dim = self.params["node_featch"] * 2 + self.params["edge_featch"] + dim * 2
                self.mlp_edge_class, _ = self.build_mlp_layer(self.params["mlp_edge_class"], dim)
            case 3:

                self.gcn_node, dim = self.build_gcn_layer(self.params["gcn_node"], dim)
                self.mlp_node_class, _ = self.build_mlp_layer(self.params["mlp_node_class"], dim)
                dim = self.params["node_featch"] * 2 + self.params["edge_featch"] + dim * 2
                self.mlp_edge_pred, dim = (None, dim) if self.params["mlp_edge_pred"] is None else self.build_mlp_layer(
                    self.params["mlp_edge_pred"], dim)
                self.gcn_edge, dim = self.build_gcn_layer(self.params["gcn_edge"], dim)
                self.mlp_edge_class, _ = self.build_mlp_layer(self.params["mlp_edge_class"], dim)

    def build_mlp_layer(self, layers, dim):
        seq = []
        for layer in layers:
            seq.append(Linear(dim, layer["out"]))

            if layer["batch_norm"]:
                seq.append(BatchNorm(layer["out"]))

            if not layer["activation"] is None:
                seq.append(self.activations[layer["activation"]]())
            dim = layer["out"]

        return torch.nn.Sequential(*seq), dim

    def build_gcn_layer(self, layers, tdim):
        module_lists = []
        dim = tdim
        for layer in layers:
            temp_list = []
            startdim = dim
            if not (layer["linear_in"] == 0):
                temp_list.append(Linear(dim, layer["linear_out"]))
                dim = layer["linear_out"]

            if layer["batch_norm"]:
                temp_list.append(BatchNorm(dim))

            if not layer["activation"] is None:
                temp_list.append(self.activations[layer["activation"]]())

            if layer["aggregation"] == "tag":
                temp_list.append(TAGConv(dim, layer["size"], K=layer["K"]))
            elif layer["aggregation"] == "gat":
                temp_list.append(GATConv(dim, layer["size"]))
            elif layer["aggregation"] == "conv":
                temp_list.append(GCNConv(dim, layer["size"]))

            module_lists.append(ModuleList(temp_list))
            if self.params['concat_gcn']:
                dim = startdim + layer['size']
            else:
                dim = layer['size']

        return ModuleList(module_lists), dim

    def build_conjugate_graph(self, A: torch.Tensor):
        A.fill_diagonal_(0)
        A = np.array(A)
        g = nx.from_numpy_array(A)
        L = nx.line_graph(g)
        A_new = torch.tensor(nx.adjacency_matrix(L).toarray())
        A_new.fill_diagonal_(1)
        return A_new.float()