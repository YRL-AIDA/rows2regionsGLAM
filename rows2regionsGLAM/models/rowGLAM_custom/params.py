PARAMS = {
    "node_featch": 15,
    "edge_featch": 4,
    "epochs": 30,
    "batch_size": 30,
    "learning_rate": 0.001,
    "model_type": 2,
    "concat_gcn": True,
    "mlp_pred": [
        {"in": -1,
         "batch_norm": True,
         "activation": "gelu",
         "out": 256},

        {"in": -1,
         "batch_norm": True,
         "activation": "gelu",
         "out": 256},
        {"in": -1,
         "batch_norm": True,
         "activation": "gelu",
         "out": 128}],
    "gcn_node":
        [
            {"linear_in": -1, # default -1: Повторить размерность прошлых слоев. 0 - не нужен линейный слой, в таком случае на Linear_out программа не обращает внимания
             "batch_norm": True,
             "activation": "gelu",
             "aggregation": "tag",
             "K": 3,  # Опционально, только проверяется только если tag
             "linear_out": 256,
             "size": 128},

            {"linear_in": -1,
             "batch_norm": True,
             "activation": "gelu",
             "K": 3,
             "aggregation": "tag",
             "linear_out": 256,
             "size": 64}
        ],

    "mlp_node_class": [
        {"in": -1,
         "batch_norm": True,
         "activation": "gelu",
         "out": 256},

        {"in": -1,
         "batch_norm": True,
         "activation": "gelu",
         "out": 128},
        {"in": -1,
         "batch_norm": True,
         "activation": "softmax",
         "out": 6}],
    "mlp_node_post": None,
    "gcn_node_post":
        [
            {"linear_in": -1,# default -1: Повторить размерность прошлых слоев. 0 - не нужен линейный слой, в таком случае на Linear_out программа не обращает внимания
             "batch_norm": True,
             "activation": "gelu",
             "aggregation": "tag",
             "K": 3,  # Опционально, только проверяется только если tag
             "linear_out": 256,
             "size": 128},

            {"linear_in": -1,
             "batch_norm": True,
             "activation": "gelu",
             "K": 3,
             "aggregation": "tag",
             "linear_out": 256,
             "size": 128}
        ],

    "mlp_node_edge":
        [
            {"in": -1,
             "batch_norm": True,
             "activation": "gelu",
             "out": 256},

            {"in": -1,
             "batch_norm": True,
             "activation": "gelu",
             "out": 128},

            {"in": -1,
             "batch_norm": True,
             "activation": None,
             "out": 128}],

    "mlp_edge_class":
        [
            {"in": -1,
             "batch_norm": True,
             "activation": "gelu",
             "out": 256},

            {"in": -1,
             "batch_norm": True,
             "activation": "gelu",
             "out": 64},

            {"in": -1,
             "batch_norm": True,
             "activation": None,
             "out": 1}],

    "seg_k": 0.5,
    "loss_params": {
        "edge_coef": 0.8,
        "node_coef": 0.2,
    },
    "sigmoidEdge": False,
    "NodeClasses": 6 # len(CLASSES)
}
