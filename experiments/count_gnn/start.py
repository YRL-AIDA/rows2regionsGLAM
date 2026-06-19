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

EPOCHS = int(os.environ.get('EPOCHS', '30'))

BASE_PARAMS = {
            "node_block": { # Первый слой содержит число features
                "linear_pred": [
                    {"in": 15, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                ],
                "gnn": [],
            },
            "node_classifier_block": {
                "linear_post": [ # Удобно задать MLP как только POST часть
                    {"in": 128+128+128, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                    {"in": 128, "out": 6, "activation": "none"},
                ]
            },
            "post_node_block": {
                "linear_pred": [
                    {"in": 128+128+128, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                ],
                "gnn": [],
                "linear_post": [
                    {"in": 128+128+128, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                    {"in": 128, "out": 128, "activation": "gelu"},
                ]
            },
            "edge_classifier_block": {
                 "linear_post": [
                    {"in": 2*(128+15)+4, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 64, "activation": "gelu"},
                    {"in": 64, "out": 1, "activation": "none"},
                ]
            },
            "epochs" : EPOCHS,
            "batch_size"  : 64,
            "learning_rate" : 0.001,
            "seg_k"  : 0.5,
            "loss_params" : {
                "edge_coef": 0.8,
                "node_coef": 0.2
            }
        }

GNN_variation = {

    1: {
        "node_block": [
            ("tag", {"batch_norm": True, "concat": True,
                     "in":128,  "in_gnn":256, "out_gnn":128, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
        ],
        "node_classifier_block": [ # Удобно задать MLP как только POST часть
            {"in": 128+128, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 128, "activation": "gelu"},
            {"in": 128, "out": 6, "activation": "none"},
        ],
        "post_node_block": [
            {"in": 128+128, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 128, "activation": "gelu"},
        ],
        
    },
    2: {
        "node_block": [
            ("tag", {"batch_norm": True, "concat": True,
                     "in":128,  "in_gnn":256, "out_gnn":128, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), 
            ("tag", {"batch_norm": False, "concat": True,
                     "in":128+128,  "in_gnn":256, "out_gnn":128, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), 
        ],
        "node_classifier_block": [ # Удобно задать MLP как только POST часть
            {"in": 128+128+128, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 128, "activation": "gelu"},
            {"in": 128, "out": 6, "activation": "none"},
        ],
        "post_node_block": [
            {"in": 128+128+128, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 128, "activation": "gelu"},
        ],
        
    },
    3: {
        "node_block": [
            ("tag", {"batch_norm": True, "concat": True,
                     "in":128,  "in_gnn":256, "out_gnn":128, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), 
            ("tag", {"batch_norm": False, "concat": True,
                     "in":128+128,  "in_gnn":256, "out_gnn":128, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), 
            ("tag", {"batch_norm": False, "concat": True,
                     "in":128+128+128,  "in_gnn":256, "out_gnn":128, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), 
        ],
        "node_classifier_block": [ # Удобно задать MLP как только POST часть
            {"in": 128+128+128+128, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 128, "activation": "gelu"},
            {"in": 128, "out": 6, "activation": "none"},
        ],
        "post_node_block": [
            {"in": 128+128+128+128, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 128, "activation": "gelu"},
        ],
        
    },
    4: {
        "node_block": [
            ("tag", {"batch_norm": True, "concat": True,
                     "in":128,  "in_gnn":256, "out_gnn":128, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), 
            ("tag", {"batch_norm": False, "concat": True,
                     "in":128+128,  "in_gnn":256, "out_gnn":128, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), 
            ("tag", {"batch_norm": False, "concat": True,
                     "in":128+128+128,  "in_gnn":256, "out_gnn":128, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), 
            ("tag", {"batch_norm": False, "concat": True,
                     "in":128+128+128+128,  "in_gnn":256, "out_gnn":128, 
                     "gnn_activation":"gelu", "activation":"gelu", "K":3}), 
        ],
        "node_classifier_block": [ # Удобно задать MLP как только POST часть
            {"in": 128+128+128+128+128, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 128, "activation": "gelu"},
            {"in": 128, "out": 6, "activation": "none"},
        ],
        "post_node_block": [
            {"in": 128+128+128+128+128, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 256, "activation": "gelu"},
            {"in": 256, "out": 128, "activation": "gelu"},
        ],
        
    },
}

if __name__ == "__main__":

    def get_exp(arg):
        params = BASE_PARAMS.copy()

        gnn_variable = GNN_variation[arg]
        params['node_block']['gnn'] = gnn_variable['node_block']
        params['node_classifier_block']['linear_post'] = gnn_variable['node_classifier_block']
        params['post_node_block']['linear_pred'] = gnn_variable['post_node_block']
        return params
        
    start_experiments(
        {f"count_gnn_{ed}": get_exp(ed)
            for ed in [1, 2, 3, 4]
        }
    ) 