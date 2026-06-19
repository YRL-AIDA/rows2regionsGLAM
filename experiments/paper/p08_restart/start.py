import os
from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch

init()

EPOCHS = int(os.environ.get("EPOCHS", "30"))

runner = ExperimentRunner("result")
runner.run({
    f"start_no_{i}": default_arch(epochs=EPOCHS, save_frequency=5)
    for i in range(10)
})
