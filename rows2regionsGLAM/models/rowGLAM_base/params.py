PARAMS_BASE  = {
    "node_featch": 15,
    "edge_featch": 4,
    "epochs": 30,
    "batch_size": 128,
    "learning_rate": 0.001,
    "Tag":[  {'in': -1, 'size': 512, 'out': 512, 'k': 3},
            {'in': 512, 'size': 256, 'out': 256, 'k': 3},
            # {'in': 256, 'size': 128, 'out': 128, 'k': 3},
            # {'in': 64, 'size': 32, 'out': 16, 'k': 3},
            ],
    "NodeLinear": [-1, 64, 32],
    "NodeLinearClassifier": [-1, 16, 8],
    "EdgeLinear": [-1, 16, 4],
    "batchNormNode": True,
    "batchNormEdge": True,
    "seg_k": 0.5,
    "loss_params": {
        "edge_coef": 0.8,
        "node_coef": 0.2,
    },
    "sigmoidEdge": False,
    "NodeClasses": 6 # len(CLASSES)
}
