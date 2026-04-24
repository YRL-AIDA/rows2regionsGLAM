PARAMS = {
    "epochs": 30,
    "batch_size": 128,
    "learning_rate": 0.001,
    "seg_k": 0.5,
    "loss_params": {
        "edge_coef": 0.8,
        "node_coef": 0.2,
    },

    "node_block": { # Первый слой содержит число features
        "linear_pred": [],
        "gnn": [
            ("tag", {"batch_norm": True, "in":15, "in_gnn":512, "out_gnn":512, 
                     "gnn_activation": "gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
            ("tag", {                    "in":512, "in_gnn":256, "out_gnn":256, 
                     "gnn_activation": "gelu", "activation":"gelu", "K":3}),
                ],
        "linear_post": [ # PRED похож на POST
            {"in": 256, "out": 64, "activation": "gelu"},
            {"in": 64,  "out": 32, "activation": "gelu"},
        ]
    },
    "node_classifier_block": {
        "linear_post": [ # Удобно задать MLP как только POST часть
            {"in": 32, "out": 16, "activation": "gelu"},
            {"in": 16, "out": 8,  "activation": "gelu"},
            {"in": 8,  "out": 6,  "activation": "softmax"},
        ]
    },
    # "post_node_block": {}, # НЕ ОБЯЗАТЕЛЬНЫЙ
    # "conjugate_edge_block": {}, # НЕ ОБЯЗАТЕЛЬНЫЙ
    "edge_classifier_block": {
         "linear_post": [ # Удобно задать MLP как только POST часть
            {"in": 98, "out": 16, "activation": "gelu"},
            {"in": 16, "out": 4,  "activation": "gelu"},
            {"in": 4,  "out": 1,  "activation": "sigmoid"},
        ]
    }
}