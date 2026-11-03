"""p15_best_restart — Stability of the best architecture from p12.

Repeats the p08_restart protocol (10 seeds, same training budget) on the
architecture that won the random search p12_arch_model: ``rand_09``.

rand_09 (reconstructed from p12 with random_sample(seed=42)):
    gnn_type=tag, num_layers=3, has_post_node=True, has_lp=True,
    hidden_dim=128, gnn_hidden=128, edge_coef=0.7 (node_coef=0.3)

Purpose: replace the default_arch numbers of p08 with the numbers of the
best architecture, so the paper reports mean ± std for the chosen model.

Run:
    bash run.sh paper/p15_best_restart --env server
"""
import os

from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch

N_SEEDS = 10

# rand_09 from p12_arch_model; loss split is stored explicitly so that
# node_coef stays exactly 0.3 (0.7 + 0.30000000000000004 otherwise).
BEST_ARCH = {
    "gnn_type": "tag",
    "num_layers": 3,
    "has_post_node": True,
    "has_lp": True,
    "hidden_dim": 128,
    "gnn_hidden": 128,
    "edge_coef": 0.7,
}
NODE_COEF = 0.3


def _build_configs(epochs):
    """Build one config per seed for the p12 winner.

    Names use the ``best_seed_*`` prefix so the runner does not reuse the
    cached checkpoints of p08 (``seed_*``) when both share ``result/``.
    """
    configs = {}
    for i in range(N_SEEDS):
        name = f"best_seed_{i}"
        params = default_arch(
            epochs=epochs,
            early_stopping_patience=3,
            seed=i,
            **BEST_ARCH,
        )
        params["loss_params"]["node_coef"] = NODE_COEF
        configs[name] = params
    return configs


def _print_config(epochs):
    print("=== p15_best_restart: p12 winner (rand_09) ===")
    for key, value in BEST_ARCH.items():
        print(f"  {key:<14s} = {value}")
    print(f"  {'node_coef':<14s} = {NODE_COEF}")
    print(f"  {'epochs':<14s} = {epochs} (early_stopping_patience=3)")
    print(f"  seeds          = 0..{N_SEEDS - 1}")


if __name__ == "__main__":
    init()
    EPOCHS = int(os.environ.get("EPOCHS", "30"))
    _print_config(EPOCHS)
    runner = ExperimentRunner("result")
    runner.run(_build_configs(EPOCHS))
