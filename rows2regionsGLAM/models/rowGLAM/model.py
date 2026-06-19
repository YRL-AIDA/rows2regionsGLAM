import torch 
from .custom_blocks import Block
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import numpy as np
import networkx as nx

from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

class TorchModel(torch.nn.Module):
    """
       Data
         |
     NodeBlock - NodeClassifierBlock -> class node
         |
    (None|PostNodeBlock)
         |
    (None|ConjugateEdgeBlock)
         |
    EdgeClassifierBlock
         |
     class edge
    """
    def __init__(self, params):
        super(TorchModel, self).__init__()
        self.node_block = Block(params["node_block"])
        self.node_classifier_block = Block(params["node_classifier_block"])
        
        self.is_exist_post_node_block = "post_node_block" in params
        if "post_node_block" in params:
            self.post_node_block = Block(params["post_node_block"])

        self.is_exist_conjugate_edge_block = "conjugate_edge_block" in params
        if "conjugate_edge_block" in params:
            self.conjugate_edge_block = Block(params["conjugate_edge_block"])
        
        self.edge_classifier_block = Block(params["edge_classifier_block"])
        

    def forward(self, data_graph_dict):
        X: torch.Tensor = data_graph_dict["X"] 
        Y: torch.Tensor = data_graph_dict["Y"]
        sp_A: torch.Tensor = data_graph_dict["sp_A"] 
        inds: List[int] = data_graph_dict["inds"]

        node_embs = self.node_block(X, sp_A)
        node_classes = self.node_classifier_block(node_embs, sp_A)
        if self.is_exist_post_node_block:
            node_embs = self.post_node_block(node_embs, sp_A)

        edge_emb = torch.cat([node_embs[inds[0]], node_embs[inds[1]], X[inds[0]], X[inds[1]], Y],dim=1)
        if self.is_exist_conjugate_edge_block:
            omega_sp_A = self.build_conjugate_graph(sp_A)
            edge_emb = self.conjugate_edge_block(edge_emb, omega_sp_A)
        else:
            omega_sp_A = None
        edge_classes = self.edge_classifier_block(edge_emb, omega_sp_A)
        edge_classes = torch.squeeze(edge_classes, 1) 
        return {
            "node_classes": node_classes, 
            "E_pred": edge_classes
        }

    # def build_conjugate_graph(self, A: torch.Tensor):
    #     # TODO optimization
    #     A.fill_diagonal_(0)
    #     A = np.array(A)
    #     g = nx.from_numpy_array(A)
    #     L = nx.line_graph(g)
    #     A_new = torch.tensor(nx.adjacency_matrix(L).toarray())
    #     A_new.fill_diagonal_(1)
    #     return A_new.float()

    def build_conjugate_graph(self, A: torch.Tensor):
        transform = LineGraph()
        if A.is_sparse:
            edge_index = A.indices().contiguous()
        else:
            edge_index = A.nonzero().t().contiguous()
        
        if edge_index.size(1) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=A.device)
        
        num_nodes = A.size(0)
        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        line_graph_data = transform(data)
        return line_graph_data.edge_index.to(A.device)
        



class CustomLoss(torch.nn.Module):
    def __init__(self, params):
        super(CustomLoss, self).__init__()
                    #BCEWithLogitsLoss
        self.bce = BCEWithLogitsLoss(pos_weight=torch.tensor(params['edge_imbalance']))
        self.ce = CrossEntropyLoss(weight=torch.tensor(params['node_imbalance']))
        self.edge_coef:float = params['edge_coef']
        self.node_coef:float = params['node_coef']

    def forward(self, pred_dict, dict_graph):
        # Ребра
        e_pred = pred_dict["E_pred"]
        e_true = dict_graph["true_edges"]
        loss_edge = self.bce(e_pred, e_true)

        # Узлы
        n_pred = pred_dict["node_classes"]
        n_true = dict_graph["true_nodes"]
        loss_node = self.ce(n_pred, n_true)

        # Строковая регуляризация
        # ang = dict_graph['Y'][:, 0]
        # sig_pred = torch.sigmoid(e_pred)
        # ang_loss = torch.dot(1-ang, 1-sig_pred)/ang.shape[0]

        loss = self.edge_coef*loss_edge  +self.node_coef*loss_node # + 0.5*ang_loss
        return loss
