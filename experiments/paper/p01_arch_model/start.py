import os
from experiments.core import init, ExperimentRunner, product
from experiments.core.model_params import default_arch

if __name__ == '__main__':
    init()
    EPOCHS = int(os.environ.get("EPOCHS", "30"))
    runner = ExperimentRunner("result")
    grid = product(
        type_model=["base", "plus_gnn"],
        conv_type=["tag", "conv", "gat"],
        lp=["lp", "nolp"],
    )
    runner.run({
        f"{t}_{c}_{l}": default_arch(
            gnn_type=c,
            has_post_node=t != "base",
            has_lp=l == "lp",
            epochs=EPOCHS,
        )
        for ((_, t), (_, c), (_, l)) in grid
    })
