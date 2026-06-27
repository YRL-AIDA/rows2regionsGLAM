import os
from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch

if __name__ == '__main__':
    init()
    EPOCHS = int(os.environ.get("EPOCHS", "30"))
    runner = ExperimentRunner("result")
    runner.run({
        f"count_gnn_{n}": default_arch(num_layers=n, epochs=EPOCHS)
        for n in [1, 2, 3, 4]
    })
