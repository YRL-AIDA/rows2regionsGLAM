"""
Block 2: Model and Data scaling experiment.

Grid: 3 data fractions × 3 hidden multipliers = 9 configurations.
All node dimensions scale proportionally via `hidden_multiplier`.
"""
import os
from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch
from rows2regionsGLAM.tokenizers.font_emb_tokenizer import RowGLAMTokenizer as EmbFontTokenizer


def scaled_arch(hidden_multiplier: int = 1, **kwargs):
    """Build architecture params with scaled dimensions.

    Args:
        hidden_multiplier: Scales hidden_dim, gnn_hidden, and node_emb_dim.
        **kwargs: Passed through to default_arch().

    Returns:
        dict: Architecture parameters dict with extra metadata keys.
    """
    params = default_arch(
        gnn_type="tag",
        num_layers=2,
        input_dim=21 + 32,  # 21 base features + 32-dim font_emb (match p02)
        epochs=int(os.environ.get("EPOCHS", "30")),
        batch_size=64,
        early_stopping_patience=3,
        seed=0,
        hidden_multiplier=hidden_multiplier,
        **kwargs,
    )
    params["_data_seed"] = 0  # deterministic subsampling
    return params


def run_block2():
    """Run Block 2: Model and Data scaling experiment (9 configs)."""
    init(os.environ.get("ENV_FILE", ".env.serverML"))

    data_fractions = [0.1, 0.5, 1.0]
    hidden_multipliers = [1, 2, 4]

    emb_font_tokenizer = EmbFontTokenizer(size=32)

    def get_tokenizer(name, params):
        return emb_font_tokenizer

    configs = {}
    for fraction in data_fractions:
        for mult in hidden_multipliers:
            name = f"ds_{int(fraction*100)}pct_mult_{mult}x"
            params = scaled_arch(hidden_multiplier=mult)
            params["_data_fraction"] = fraction
            configs[name] = params

    runner = ExperimentRunner("result", get_tokenizer=get_tokenizer)
    runner.run(configs)

    # Print configuration summary
    print("\n=== Block 2: Model and Data Scaling — Configuration Summary ===")
    print(f"{'Config':<25s} {'hidden_dim':>10s} {'gnn_hidden':>10s} {'node_emb':>10s} {'cls_hidden':>18s} {'data_pct':>10s}")
    print("-" * 90)
    for fraction in data_fractions:
        for mult in hidden_multipliers:
            name = f"ds_{int(fraction*100)}pct_mult_{mult}x"
            hd = 128 * mult
            gh = 256 * mult
            ne = 64 * mult
            ch = f"[{ne*4}, {ne*2}]"
            pct = f"{int(fraction*100)}%"
            print(f"  {name:<23s} {hd:>10d} {gh:>10d} {ne:>10d} {ch:>18s} {pct:>10s}")


if __name__ == "__main__":
    run_block2()
