import os
from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch

if __name__ == '__main__':
    init()
    EPOCHS = int(os.environ.get("EPOCHS", "30"))
    runner = ExperimentRunner("result")
    runner.run({
        f"lr_{lr:6.4f}_bs_{bs}": default_arch(
            learning_rate=lr,
            batch_size=bs,
            epochs=EPOCHS,
        )
        for lr in [0.01, 0.005, 0.001, 0.0005]
        for bs in [1, 8, 64, 128]
    })
