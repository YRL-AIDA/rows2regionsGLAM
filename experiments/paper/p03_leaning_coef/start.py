import os
from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch

init()

EPOCHS = int(os.environ.get("EPOCHS", "30"))

runner = ExperimentRunner("result")
runner.run({
    f"ed_coef{ec:5.2f}": default_arch(
        edge_coef=ec,
        epochs=EPOCHS,
        save_frequency=5,
    )
    for ec in [0.0, 0.2, 0.5, 0.8, 1.0]
})
