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
                "gnn": [
                    ("tag", {"batch_norm": True, "concat": True,
                             "in":128,  "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
                    ("tag", {"batch_norm": False,"concat": True,
                             "in":128+128, "in_gnn":256, "out_gnn":128, 
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
                "gnn": [
                    ("tag", {"batch_norm": True, "concat": True,
                             "in":128,  "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
                    ("tag", {"batch_norm": False,"concat": True,
                             "in":128+128, "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                        ],
                "linear_post": [ # PRED похож на POST
                    {"in": 128+128+128, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                    {"in": 128, "out": 128, "activation": "gelu"},
                ]
            }, # НЕ ОБЯЗАТЕЛЬНЫЙ
            # "conjugate_edge_block": {}, # НЕ ОБЯЗАТЕЛЬНЫЙ
            "edge_classifier_block": {
                 "linear_post": [ # Удобно задать MLP как только POST часть
                    {"in": 2*(128+15)+4, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 64, "activation": "gelu"},
                    {"in": 64, "out": 1, "activation": "none"},
                ]
            },
            "epochs" : EPOCHS,
            "batch_size"  : 64,
            "learning_rate" : 0.001,
            "seg_k"  : 0.5,
            "save_frequency": 5,
            "loss_params" : {
                "edge_coef": 0.8,
                "node_coef": 0.2
            },
            
        }


if __name__ == "__main__":

    def get_exp(arg):
        params = BASE_PARAMS.copy()
        params["loss_params"] = {
            "edge_coef": arg,
            "node_coef": 1-arg
        }
        return params
        
    start_experiments(
        {f"ed_coef{ed:5.2f}": get_exp(ed)
            # for ed in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
            for ed in [0.0, 0.2, 0.5, 0.8, 1.0]
        }
    ) 