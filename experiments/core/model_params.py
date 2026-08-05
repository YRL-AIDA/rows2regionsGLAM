def mlp(in_dim, hidden_sizes, out_dim, activation="gelu", last_activation="none"):
    layers = []
    prev = in_dim
    for size in hidden_sizes:
        layers.append({"in": prev, "out": size, "activation": activation})
        prev = size
    layers.append({"in": prev, "out": out_dim, "activation": last_activation})
    return layers


def tag_gnn(in_dim, num_layers, hidden_dim=128, gnn_hidden=256, gnn_type="tag"):
    layers = []
    for i in range(num_layers):
        in_d = in_dim if i == 0 else in_dim + i * hidden_dim
        batch_norm = (i == 0)
        layers.append((gnn_type, {
            "batch_norm": batch_norm,
            "concat": True,
            "in": in_d,
            "in_gnn": gnn_hidden,
            "out_gnn": hidden_dim,
            "gnn_activation": "gelu",
            "activation": "gelu",
            "K": 3,
        }))
    return layers


def node_classifier(in_dim, num_classes, hidden_sizes=None, node_emb_dim=None):
    if hidden_sizes is None:
        if node_emb_dim is not None:
            hidden_sizes = [node_emb_dim * 4, node_emb_dim * 2]
        else:
            hidden_sizes = [256, 128]
    return mlp(in_dim, hidden_sizes, num_classes, activation="gelu", last_activation="none")


def edge_classifier(in_dim, hidden_sizes=None, node_emb_dim=None):
    if hidden_sizes is None:
        if node_emb_dim is not None:
            hidden_sizes = [node_emb_dim * 4, node_emb_dim]
        else:
            hidden_sizes = [256, 64]
    return mlp(in_dim, hidden_sizes, 1, activation="gelu", last_activation="none")


def default_arch(
    gnn_type="tag",
    num_layers=2,
    has_post_node=True,
    has_lp=True,
    input_dim=53,
    num_classes=6,
    edge_dim=4,
    hidden_dim=128,
    gnn_hidden=256,
    epochs=30,
    batch_size=64,
    learning_rate=0.001,
    seg_k=0.5,
    edge_coef=0.8,
    save_frequency=None,
    seed=None,
    early_stopping_patience=None,
    hidden_multiplier=1,
    node_emb_dim=None,
):
    # Apply hidden multiplier — scales all node dimensions proportionally
    if hidden_multiplier != 1:
        hidden_dim = 128 * hidden_multiplier
        gnn_hidden = 256 * hidden_multiplier
    if node_emb_dim is None:
        node_emb_dim = 64 * hidden_multiplier

    node_coef = 1.0 - edge_coef
    gnn_first_in = hidden_dim if has_lp else input_dim

    node_gnn = tag_gnn(gnn_first_in, num_layers, hidden_dim=hidden_dim, gnn_hidden=gnn_hidden, gnn_type=gnn_type)
    concat_node_out = (hidden_dim if has_lp else input_dim) + num_layers * hidden_dim

    params = {
        "node_block": {
            "linear_pred": [
                {"in": input_dim, "out": 256, "activation": "gelu"},
                {"in": 256, "out": hidden_dim, "activation": "gelu"},
            ] if has_lp else [],
            "gnn": node_gnn,
        },
        "node_classifier_block": {
            "linear_post": node_classifier(concat_node_out, num_classes, node_emb_dim=node_emb_dim),
        },
        "edge_classifier_block": {
            "linear_post": edge_classifier(2 * (hidden_dim + input_dim) + edge_dim, node_emb_dim=node_emb_dim)
            if has_post_node
            else edge_classifier(2 * (concat_node_out + input_dim) + edge_dim, node_emb_dim=node_emb_dim),
        },
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seg_k": seg_k,
        "loss_params": {
            "edge_coef": edge_coef,
            "node_coef": node_coef,
        },
    }

    if save_frequency is not None:
        params["save_frequency"] = save_frequency

    if seed is not None:
        params["seed"] = seed

    if early_stopping_patience is not None:
        params["early_stopping_patience"] = early_stopping_patience

    if has_post_node:
        post_concat_in = concat_node_out
        post_gnn_out = hidden_dim + num_layers * hidden_dim
        post_gnn = tag_gnn(hidden_dim, num_layers, hidden_dim=hidden_dim, gnn_hidden=gnn_hidden, gnn_type=gnn_type)
        params["post_node_block"] = {
            "linear_pred": [
                {"in": post_concat_in, "out": 256, "activation": "gelu"},
                {"in": 256, "out": 256, "activation": "gelu"},
                {"in": 256, "out": hidden_dim, "activation": "gelu"},
            ],
            "gnn": post_gnn,
            "linear_post": [
                {"in": post_gnn_out, "out": 256, "activation": "gelu"},
                {"in": 256, "out": hidden_dim, "activation": "gelu"},
                {"in": hidden_dim, "out": hidden_dim, "activation": "gelu"},
            ],
        }

    return params


def model_type_base_params(input_dim=15, num_classes=6):
    return {
        "edge_featch": 4,
        "node_featch": input_dim,
        "Tag": [
            {"in": input_dim, "size": 512, "out": 512, "k": 3},
            {"in": 512, "size": 256, "out": 256, "k": 3},
        ],
        "NodeLinear": [271, 64, 32],
        "NodeLinearClassifier": [32, 16, 8],
        "EdgeLinear": [98, 16, 4],
        "batchNormNode": True,
        "batchNormEdge": True,
        "sigmoidEdge": False,
        "NodeClasses": num_classes,
    }


def model_type_custom_params(input_dim=15, num_classes=6):
    return {
        "edge_featch": 4,
        "node_featch": input_dim,
        "model_type": 2,
        "concat_gcn": True,
        "mlp_pred": [
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 256},
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 128},
        ],
        "gcn_node": [
            {"linear_in": -1, "linear_out": 256, "batch_norm": True, "activation": "gelu", "aggregation": "tag", "K": 3, "size": 128},
            {"linear_in": -1, "linear_out": 256, "batch_norm": False, "activation": "gelu", "aggregation": "tag", "K": 3, "size": 64},
        ],
        "mlp_node_class": [
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 256},
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 128},
            {"in": -1, "batch_norm": False, "activation": "softmax", "out": num_classes},
        ],
        "mlp_node_pred": [
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 256},
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 256},
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 128},
        ],
        "gcn_node_post": [
            {"linear_in": -1, "linear_out": 256, "batch_norm": True, "activation": "gelu", "aggregation": "tag", "K": 3, "size": 128},
            {"linear_in": -1, "linear_out": 256, "batch_norm": False, "activation": "gelu", "aggregation": "tag", "K": 3, "size": 128},
        ],
        "mlp_node_edge": [
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 256},
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 128},
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 128},
        ],
        "mlp_edge_class": [
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 256},
            {"in": -1, "batch_norm": False, "activation": "gelu", "out": 64},
            {"in": -1, "batch_norm": False, "activation": None, "out": 1},
        ],
        "sigmoidEdge": False,
        "NodeClasses": num_classes,
    }
