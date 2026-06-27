import os
from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch

if __name__ == '__main__':
    init()
    EPOCHS = int(os.environ.get("EPOCHS", "30"))
    runner = ExperimentRunner("result")
    runner.run({
        f"seed_{i}": default_arch(epochs=EPOCHS, early_stopping_patience=3, seed=i)
        for i in range(10)
    })
