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
        mlp0 = 15
        mlp1 = int(coef*256)
        mlp2 = int(coef*512)         # Выход
        
        mlp1_in = mlp2 
        gnn1_in = int(coef*1024)
        gnn1_out = int(coef*512)      # Выход
        
        mlp2_in = mlp2+gnn1_out
        gnn2_in = int(coef*1024)
        gnn2_out = int(coef*512)      # Выход
        
        cls_1 = mlp1_in+gnn1_out+gnn2_out
        cls_2 = int(coef*512)
        cls_3 = int(coef*64)
        
        mlp1_seg = mlp1_in+gnn1_out+gnn2_out
        mlp2_seg = int(coef*1024)
        mlp3_seg = int(coef*512)
        mlp4_seg = int(coef*256)
        
        mlp3_in = mlp4_seg
        gnn3_in = int(coef*256)
        gnn3_out = int(coef*256)
        
        mlp4_in = mlp4_seg+gnn3_out
        gnn4_in = int(coef*256)
        gnn4_out = int(coef*128)
        
        seg1 = 2*(mlp4_seg+gnn3_out+gnn4_out+mlp0)+4
        seg2 = int(coef*128)
        seg3 = int(coef*32)

        BASE_PARAMS = {
    "node_block": { # Первый слой содержит число features
        "linear_pred": [
            {"in": mlp0, "out": mlp1, "activation": "gelu"},
            {"in": mlp1, "out": mlp2, "activation": "gelu"},
        ],
        "gnn": [
            ("tag", {"batch_norm": True, "concat": True,
                     "in":mlp1_in, "in_gnn":gnn1_in, "out_gnn":gnn1_out, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
            ("tag", {"batch_norm": False,"concat": True,
                     "in":mlp2_in, "in_gnn":gnn2_in, "out_gnn":gnn2_out, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                ],
    },
    "node_classifier_block": {
        "linear_post": [ # Удобно задать MLP как только POST часть
            {"in": cls_1, "out": cls_2, "activation": "gelu"},
            {"in": cls_2, "out": cls_3, "activation": "gelu"},
            {"in": cls_3, "out": 6,     "activation": "none"},
        ]
    },
    "post_node_block": {
        "linear_pred": [
            {"in": mlp1_seg, "out": mlp2_seg, "activation": "gelu"},
            {"in": mlp2_seg, "out": mlp3_seg, "activation": "gelu"},
            {"in": mlp3_seg, "out": mlp4_seg, "activation": "gelu"},
        ],
        "gnn": [
            ("tag", {"batch_norm": True, "concat": True,
                     "in":mlp3_in,  "in_gnn":gnn3_in, "out_gnn":gnn3_out, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
            ("tag", {"batch_norm": False,"concat": True,
                     "in":mlp4_in, "in_gnn":gnn4_in, "out_gnn":gnn4_out, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                ],
    }, 
    "edge_classifier_block": {
         "linear_post": [ # Удобно задать MLP как только POST часть
            {"in": seg1, "out": seg2, "activation": "gelu"},
            {"in": seg2, "out": seg3, "activation": "gelu"},
            {"in": seg3, "out": 1, "activation": "none"},
        ]
    },
    "epochs" : 10,
    "batch_size"  : 64,
    "learning_rate" : 0.001,
    "seg_k"  : 0.5,
    "save_frequency": 5,
    "loss_params" : {
        "edge_coef": 0.8,
        "node_coef": 0.2
    },
    
}

        return BASE_PARAMS
        
    start_experiments(
        {f"size_times_{coef}_no{ed}": get_exp(coef)
            for coef in [0.125, 0.25, 0.5, 1, 2, 4] for ed in range(3)
        }
    ) 