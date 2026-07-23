"""
Block 2: Model and Data scaling experiment.

Grid: 3 dataset sizes × 3 hidden multipliers = 9 configurations.
All node dimensions scale proportionally via `hidden_multiplier`.
"""
import os
from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch
from rows2regionsGLAM.tokenizers.font_emb_tokenizer import RowGLAMTokenizer as EmbFontTokenizer


def scaled_arch(hidden_multiplier: int = 1, dataset_fraction: float = 1.0, **kwargs):
    """Build architecture params with scaled dimensions.

    Args:
        hidden_multiplier: Scales hidden_dim, gnn_hidden, and node_emb_dim.
        dataset_fraction: Fraction of total training data to use.
        **kwargs: Passed through to default_arch().

    Returns:
        dict: Architecture parameters dict with extra metadata keys.
    """
    params = default_arch(
        gnn_type="tag",
        num_layers=2,
        input_dim=21 + 512,  # 21 base features + 512-dim font_emb (match p02)
        epochs=int(os.environ.get("EPOCHS", "30")),
        batch_size=64,
        early_stopping_patience=3,
        seed=0,
        hidden_multiplier=hidden_multiplier,
        **kwargs,
    )
    params["_data_fraction"] = dataset_fraction
    params["_data_seed"] = 0  # deterministic subsampling
    params["_cache_dir"] = f"tmp_scale_{int(dataset_fraction * 300000)}_x{hidden_multiplier}"
    return params


def run_block2():
    """Run Block 2: Model and Data scaling experiment (9 configs)."""
    init(os.environ.get("ENV_FILE", ".env.serverML"))

    # Total training rows in PubLayNet ≈ 300k
    TOTAL_ROWS = 300000
    dataset_sizes = [10000, 50000, 300000]
    hidden_multipliers = [1, 2, 4]

    emb_font_tokenizer = EmbFontTokenizer(size=512)

    def get_tokenizer(name, params):
        return emb_font_tokenizer

    configs = {}
    for ds_size in dataset_sizes:
        fraction = ds_size / TOTAL_ROWS
        for mult in hidden_multipliers:
            name = f"ds_{ds_size}_mult_{mult}x"
            configs[name] = scaled_arch(
                hidden_multiplier=mult,
                dataset_fraction=fraction,
            )

    runner = ExperimentRunner("result", get_tokenizer=get_tokenizer)
    runner.run(configs)

    # Print configuration summary
    print("\n=== Block 2: Model and Data Scaling — Configuration Summary ===")
    print(f"{'Config':<25s} {'hidden_dim':>10s} {'gnn_hidden':>10s} {'node_emb':>10s} {'cls_hidden':>18s} {'data_rows':>10s}")
    print("-" * 90)
    for ds_size in dataset_sizes:
        for mult in hidden_multipliers:
            name = f"ds_{ds_size}_mult_{mult}x"
            hd = 128 * mult
            gh = 256 * mult
            ne = 64 * mult
            ch = f"[{ne*4}, {ne*2}]"
            print(f"  {name:<23s} {hd:>10d} {gh:>10d} {ne:>10d} {ch:>18s} {ds_size:>10,d}")


if __name__ == "__main__":
    run_block2()
