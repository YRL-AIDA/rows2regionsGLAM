"""
Block 3: Random Architecture Search — p12_arch_model.

Sample 24 random architectures from the parameter space and evaluate each
with identical training conditions (seed=0, font_emb tokenizer, 32-dim).

Usage:
    python experiments/paper/p12_arch_model/start.py
"""
import os
import sys
from experiments.core import init, ExperimentRunner, random_sample
from experiments.core.model_params import default_arch
from rows2regionsGLAM.tokenizers.font_emb_tokenizer import RowGLAMTokenizer as EmbFontTokenizer


def _build_configs():
    """Sample 24 random architecture configurations.

    Returns:
        list[tuple[str, dict]]: (name, params) pairs.
    """
    fixed = {
        "input_dim": 21 + 32,
        "epochs": int(os.environ.get("EPOCHS", "30")),
        "batch_size": 64,
        "early_stopping_patience": 3,
        "seed": 0,
    }

    axes = {
        "gnn_type": ["tag", "conv", "gat"],
        "num_layers": [1, 2, 3, 4],
        "edge_coef": (0.0, 1.0),
        "has_post_node": [True, False],
        "has_lp": [True, False],
        "hidden_dim": [64, 128, 256, 512],
        "gnn_hidden": [128, 256, 512],
    }

    samples = random_sample(seed=42, n_configs=24, **axes)

    configs = {}
    for i, samp in enumerate(samples):
        name = f"rand_{i:02d}"
        params = default_arch(
            gnn_type=samp["gnn_type"],
            num_layers=samp["num_layers"],
            has_post_node=samp["has_post_node"],
            has_lp=samp["has_lp"],
            hidden_dim=samp["hidden_dim"],
            gnn_hidden=samp["gnn_hidden"],
            edge_coef=samp["edge_coef"],
            **fixed,
        )
        # Ensure loss_params are consistent (default_arch does this, but
        # we keep the explicit set as a safeguard).
        params.setdefault("loss_params", {})["edge_coef"] = samp["edge_coef"]
        params["loss_params"]["node_coef"] = round(1.0 - samp["edge_coef"], 2)
        configs[name] = params

    return configs


def _print_summary(configs):
    """Print a summary table of all sampled configurations."""
    header = (
        f"{'Config':<12s} {'gnn_type':>8s} {'num_layers':>10s} "
        f"{'edge_coef':>10s} {'has_post_node':>14s} {'has_lp':>8s} "
        f"{'hidden_dim':>10s} {'gnn_hidden':>10s}"
    )
    sep = "-" * len(header)

    print(f"\n=== Block 3: Random Architecture Search — Configuration Summary ===")
    print(f"Total unique configs: {len(configs)}")
    print(sep)
    print(header)
    print(sep)

    # Reconstruct original sampled params from the config name order
    for name, params in configs.items():
        lp = params["loss_params"]
        has_post = "post_node_block" in params
        has_lp = len(params["node_block"].get("linear_pred", [])) > 0
        gnn_type = params["node_block"]["gnn"][0][0]
        num_layers = len(params["node_block"]["gnn"])
        hidden_dim = params["node_block"]["gnn"][0][1].get("out_gnn", "?")
        gnn_hidden = params["node_block"]["gnn"][0][1].get("in_gnn", "?")
        edge_coef = lp.get("edge_coef", "?")

        print(
            f"  {name:<10s} {gnn_type:>8s} {num_layers:>10d} "
            f"{str(edge_coef):>10s} {str(has_post):>14s} {str(has_lp):>8s} "
            f"{str(hidden_dim):>10s} {str(gnn_hidden):>10s}"
        )

    print(sep)


def run():
    """Run Block 3: Random Architecture Search experiment."""
    init(os.environ.get("ENV_FILE", ".env.serverML"))

    emb_font_tokenizer = EmbFontTokenizer(size=32)

    def get_tokenizer(name, params):
        return emb_font_tokenizer

    configs = _build_configs()

    runner = ExperimentRunner("result", get_tokenizer=get_tokenizer)
    runner.run(configs)

    _print_summary(configs)


if __name__ == "__main__":
    run()
