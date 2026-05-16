import warnings
import sys, os
import torch
from pathlib import Path
from dotenv import load_dotenv

PATH_PROJECT = os.path.join('..', '..')
sys.path.append(PATH_PROJECT)
env_file = os.path.join(PATH_PROJECT, '.env')
load_dotenv(env_file)
warnings.filterwarnings('ignore', message='Converting sparse tensor to CSR format')
warnings.filterwarnings('ignore', message='Implicit dimension choice for softmax')
from utils import start_experiments 

if __name__ == "__main__":

    def get_exp(coef):
        
        input_x = 15
        input_y = 4
        
        linear1_pred = [int(coef*256), int(coef*128)]
        gnn1 = [(int(coef*256), int(coef*128)), (int(coef*256), int(coef*128))]
        
        cl_node = [int(coef*256), int(coef*128)]
        
        linear2_pred = [int(coef*256), int(coef*128)]
        gnn2 = [(int(coef*256), int(coef*128)), (int(coef*256), int(coef*128))]
        linear2_post = [int(coef*256), int(coef*128)]

        edge = [int(coef*256), int(coef*64)]

        
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
                # "linear_post": [ # PRED похож на POST
                #     {"in": 128+128+128, "out": 256, "activation": "gelu"},
                #     {"in": 256, "out": 128, "activation": "gelu"},
                #     {"in": 128, "out": 32, "activation": "gelu"},
                # ]
            },
            "node_classifier_block": {
                "linear_post": [ # Удобно задать MLP как только POST часть
                    {"in": linear1_pred[-1]+gnn1[0][1]+gnn1[1][1], "out": cl_node[0], "activation": "gelu"},
                    {"in": cl_node[0], "out": cl_node[1], "activation": "gelu"},
                    {"in": cl_node[1], "out": 6, "activation": "none"},
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
                    {"in": 2*(linear2_post[1]+input_x)+input_y, "out": edge[0], "activation": "gelu"},
                    {"in": edge[0], "out": edge[1], "activation": "gelu"},
                    {"in": edge[1], "out": 1, "activation": "none"},
                ]
            },
            "epochs" : 10,
            "batch_size"  : 64,
            "learning_rate" : 0.001,
            "seg_k"  : 0.5,
            "loss_params" : {
                "edge_coef": 0.8,
                "node_coef": 0.2
            }
        }

        return params
        
    start_experiments(
        {f"size_times_{coef}": get_exp(coef)
            for coef in [0.25, 0.5, 1, 2, 4]
        }
    ) 