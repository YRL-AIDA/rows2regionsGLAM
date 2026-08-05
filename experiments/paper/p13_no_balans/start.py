import copy
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch
from rows2regionsGLAM.utils.imbalance import calculate_imbalance
from rows2regionsGLAM.utils.trainer import Trainer
from rows2regionsGLAM.models import get_loss, get_model, save_model


class NoBalanceRunner(ExperimentRunner):
    """
    ExperimentRunner that controls class imbalance handling via _balancing.

    _balancing values:
        "both"  — node + edge imbalance (default behaviour)
        "edges" — edge imbalance only, uniform node weights
        "nodes" — node imbalance only, no edge pos_weight
        "none"  — no imbalance, uniform weights everywhere
    """

    def _train(self, model_info, train_dataset, model_params, val_dataset=None):
        model_name = model_info["model_name"]
        params = copy.deepcopy(model_info["model_params"])

        train_dataset.train()

        balancing = model_params.get("_balancing", "both")
        num_classes = len(self._coco_manager_train.classes)

        if balancing in ("both", "edges", "nodes"):
            node_imbalance, edge_imbalance = calculate_imbalance(train_dataset)
            if any(math.isnan(v) for v in node_imbalance):
                node_imbalance = [1.0] * num_classes
            if math.isnan(edge_imbalance):
                edge_imbalance = 1.0
        else:
            node_imbalance = [1.0] * num_classes
            edge_imbalance = 1.0

        if balancing in ("edges", "none"):
            node_imbalance = [1.0] * num_classes
        if balancing in ("nodes", "none"):
            edge_imbalance = 1.0

        loss_params = params.setdefault("loss_params", {})
        loss_params["node_imbalance"] = node_imbalance
        loss_params["edge_imbalance"] = edge_imbalance

        if Path(model_name).exists():
            return

        params["node_classifier_block"]["linear_post"][-1]["activation"] = "none"
        params["edge_classifier_block"]["linear_post"][-1]["activation"] = "none"

        seed = params.get("seed")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        model, num_restart = get_model(params, model_name)
        params["restart_num"] = num_restart
        loss = get_loss(params["loss_params"])
        trainer = Trainer(model=model, dataset=train_dataset, val_dataset=val_dataset,
                          loss=loss, train_param=params, loger=self.loger)
        trainer.start_train()
        model = trainer.model
        save_model(model, model_name)


if __name__ == '__main__':
    init()
    EPOCHS = int(os.environ.get("EPOCHS", "30"))

    runner = NoBalanceRunner("result")

    configs = {}
    for balancing in ["both", "edges", "nodes", "none"]:
        for seed in range(3):
            name = f"{balancing}_seed_{seed}"
            arch = default_arch(
                epochs=EPOCHS,
                early_stopping_patience=3,
                seed=seed,
            )
            arch["_balancing"] = balancing
            configs[name] = arch

    runner.run(configs)
