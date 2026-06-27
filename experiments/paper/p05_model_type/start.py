import os
from experiments.core import init, ExperimentRunner
from experiments.core.model_params import (
    default_arch,
    model_type_base_params,
    model_type_custom_params,
)

if __name__ == '__main__':
    init()
    EPOCHS = int(os.environ.get("EPOCHS", "30"))

    def add_training_params(params):
        p = dict(params)
        p["epochs"] = EPOCHS
        p["batch_size"] = 64
        p["learning_rate"] = 0.001
        p["seg_k"] = 0.5
        p.setdefault("loss_params", {"edge_coef": 0.8, "node_coef": 0.2})
        return p

    runner = ExperimentRunner("result")
    runner.run({
        "main": default_arch(epochs=EPOCHS),
    })
