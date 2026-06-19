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
            "epochs" : EPOCHS,
            "batch_size"  : 64,
            "learning_rate" : 0.001,
            "seg_k"  : 0.5,
            "loss_params" : {
                "edge_coef": 0.8,
                "node_coef": 0.2
            }
        }



if __name__ == "__main__":

    def get_exp(type_model, conv_type, lp):
        params = BASE_PARAMS.copy()
        lp = lp == 'lp'
        input_gnn = 128 if lp else 15
        
        params["node_block"] = { # Первый слой содержит число features
                "linear_pred": [
                    {"in": 15, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                ] if lp else [],
                "gnn": [
                    (conv_type, {"batch_norm": True, "concat": True,
                             "in": input_gnn,  "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
                    (conv_type, {"batch_norm": False,"concat": True,
                             "in":128+input_gnn, "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                        ],
        }
        
        params["node_classifier_block"] = {
                "linear_post": [ # Удобно задать MLP как только POST часть
                    {"in": 128+128+input_gnn, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                    {"in": 128, "out": 6, "activation": "none"},
                ]
        }

        if type_model != 'base':
            input_2gnn = 128 if lp else 128+128+input_gnn
            params["post_node_block"] = {
                "linear_pred": [
                    {"in": 128+128+input_gnn, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                ] if lp else [],
                "gnn": [
                    (conv_type, {"batch_norm": True, "concat": True,
                             "in":input_2gnn ,  "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
                    (conv_type, {"batch_norm": False,"concat": True,
                             "in":128+input_2gnn, "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                        ],
                "linear_post": [ # PRED похож на POST
                    {"in": 128+128+input_2gnn, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                    {"in": 128, "out": 128, "activation": "gelu"},
                ]
            }
        if type_model == 'conjugate':
            input_3gnn = 2*(128+15)+4
            params["conjugate_edge_block"] = {
                "gnn": [
                    (conv_type, {"batch_norm": True, "concat": True,
                             "in":input_3gnn,  "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
                    (conv_type, {"batch_norm": False,"concat": True,
                             "in":input_3gnn+128, "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                        ],
            }

            
        if type_model == 'base':
            edge_input = 2*(128+128+input_gnn+15)+4
        elif type_model == 'plus_gnn':
            edge_input = 2*(128+15)+4
        elif type_model == 'conjugate':
            edge_input = input_3gnn+128+128
        params["edge_classifier_block"] = {
                 "linear_post": [ # Удобно задать MLP как только POST часть
                    {"in": edge_input, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 64, "activation": "gelu"},
                    {"in": 64, "out": 1, "activation": "none"},
                ]
        }
     


        
        return params
        
    start_experiments(
        {f"{type_model}_{conv_type}_{lp}": get_exp(type_model, conv_type, lp)
            for type_model in ["base", "plus_gnn"] for conv_type in ["tag", "conv", "gat"] for lp in ["lp", "nolp"] #, "conjugate" - не работает из-за устройств (надо через торч строить)
        }
    ) 