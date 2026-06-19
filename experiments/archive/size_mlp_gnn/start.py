import warnings
import sys, os
import torch
from pathlib import Path
from dotenv import load_dotenv
import itertools
PATH_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PATH_PROJECT)
env_file = os.path.join(PATH_PROJECT, '.env')
load_dotenv(env_file)
warnings.filterwarnings('ignore', message='Converting sparse tensor to CSR format')
warnings.filterwarnings('ignore', message='Implicit dimension choice for softmax')
from utils import start_experiments 

if __name__ == "__main__":

    def get_exp(coef_gnn, coef_class):
        
        input_x = 15
        input_y = 4
        
        linear1_pred = [int(coef_gnn*256), int(coef_gnn*128)]
        gnn1 = [(int(coef_gnn*256), int(coef_gnn*128)), (int(coef_gnn*256), int(coef_gnn*128))]
        
        cl_node = [int(coef_gnn*256), int(coef_gnn*128)]
        
        linear2_pred = [int(coef_gnn*256), int(coef_gnn*128)]
        gnn2 = [(int(coef_gnn*256), int(coef_gnn*128)), (int(coef_gnn*256), int(coef_gnn*128))]
        linear2_post = [int(coef_gnn*256), int(coef_gnn*128)]

        edge = [int(coef_gnn*256), int(coef_gnn*64)]

        
        params = {
            "node_block": { # Первый слой содержит число features
                "linear_pred": [
                    {"in": input_x, "out": linear1_pred[0], "activation": "gelu"},
                    {"in": linear1_pred[0], "out": linear1_pred[1], "activation": "gelu"},
                ],
                "gnn": [
                    ("tag", {"batch_norm": True, "concat": True,
                             "in":linear1_pred[-1],  "in_gnn":gnn1[0][0], "out_gnn":gnn1[0][1], 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
                    ("tag", {"batch_norm": False,"concat": True,
                             "in":linear1_pred[-1]+gnn1[0][1], "in_gnn":gnn1[1][0], "out_gnn":gnn1[1][1], 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                        ],

            },
            "node_classifier_block": {
                "linear_post": [ # Удобно задать MLP как только POST часть
                    {"in": linear1_pred[-1]+gnn1[0][1]+gnn1[1][1], "out": int(cl_node[0]*coef_class), "activation": "gelu"},
                    {"in": int(cl_node[0]*coef_class), "out": int(cl_node[1]*coef_class), "activation": "gelu"},
                    {"in": int(cl_node[1]*coef_class), "out": 6, "activation": "none"},
                ]
            },
            "post_node_block": {
                "linear_pred": [
                    {"in": linear1_pred[-1]+gnn1[0][1]+gnn1[1][1], "out": linear2_pred[0], "activation": "gelu"},
                    {"in": linear2_pred[0], "out": linear2_pred[1], "activation": "gelu"},
                ],
                "gnn": [
                    ("tag", {"batch_norm": True, "concat": True,
                             "in":linear2_pred[-1],  "in_gnn":gnn2[0][0], "out_gnn":gnn2[0][1], 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
                    ("tag", {"batch_norm": False,"concat": True,
                             "in":linear2_pred[-1]+gnn2[0][1], "in_gnn":gnn2[1][0], "out_gnn":gnn2[1][1], 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                        ],
                "linear_post": [ # PRED похож на POST
                    {"in": linear2_pred[-1]+gnn2[0][1]+gnn2[1][1], "out": linear2_post[0], "activation": "gelu"},
                    {"in": linear2_post[0], "out": linear2_post[1], "activation": "gelu"},
                ]
            }, # НЕ ОБЯЗАТЕЛЬНЫЙ
            # "conjugate_edge_block": {}, # НЕ ОБЯЗАТЕЛЬНЫЙ
            "edge_classifier_block": {
                 "linear_post": [ # Удобно задать MLP как только POST часть
                    {"in": 2*(linear2_post[1]+input_x)+input_y, "out": int(edge[0]*coef_class), "activation": "gelu"},
                    {"in": int(edge[0]*coef_class), "out": int(edge[1]*coef_class), "activation": "gelu"},
                    {"in": int(edge[1]*coef_class), "out": 1, "activation": "none"},
                ]
            },
            "epochs" : 10,
            "batch_size"  : 64,
            "learning_rate" : 0.001,
            "seg_k"  : 0.5,
            "loss_params" : {
                "edge_coef": 0.8,
                "node_coef": 0.2
            }, 
            "save_frequency":10,
            "restart_num":3

        }

        return params
    # Эксперимент 2   
    # coefs_class = [1]
    # coefs_gnn = [8, 12, 16]

    # Эксперимент 3
    coefs_class = [1]
    coefs_gnn = [4, 8, 12, 16]
    exps = list(itertools.product(coefs_class, coefs_gnn))
    print(exps)
    start_experiments(
        {f"mlp {coef_mlp} gnn{coef_gnn}": get_exp(coef_gnn, coef_mlp)
            for coef_mlp, coef_gnn in exps
        }
    ) 