import copy
import json
import math
import os
from pathlib import Path

from rows2regionsGLAM.utils.loger import Loger
from rows2regionsGLAM.utils.coco_manager import COCOManager
from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.datasetloaders.base_line_dataset import GLAMDataset
from rows2regionsGLAM.utils.trainer import Trainer
from rows2regionsGLAM.utils.tester import Tester
from rows2regionsGLAM.utils.imbalance import calculate_imbalance
from rows2regionsGLAM.tokenizers import RowGLAMTokenizer
from rows2regionsGLAM.pipeline import Pipeline
from rows2regionsGLAM.models import get_loss, get_model, save_model


class ExperimentRunner:
    def __init__(self, result_path, model_factory=None, get_tokenizer=None, tester_factory=None):
        self.result_path = Path(result_path)
        if not self.result_path.exists():
            self.result_path.mkdir(parents=True)

        self.loger = Loger()
        self._model_factory = model_factory
        self._tester_factory = tester_factory

        default_tokenizer_factory = lambda name, params: RowGLAMTokenizer()
        self._get_tokenizer = get_tokenizer or default_tokenizer_factory

        self._train_dataset_name = os.environ["NAME_DATASET"]
        self._train_dataset_path = os.environ["DATASET_PATH"]
        self._train_dataset_coco = os.environ["COCO_PATH"]

        self._test_dataset_name = os.environ["NAME_TEST_DATASET"]
        self._test_dataset_path = os.environ["TEST_PATH"]
        self._test_dataset_coco = os.environ["TEST_COCO_PATH"]

        self._cache_pdf = os.environ["CASH_PDF_PATH"]

        self._coco_manager_train = COCOManager(
            loger=self.loger,
            coco_path=self._train_dataset_coco,
            name_dataset=self._train_dataset_name,
        )
        self._coco_manager_test = COCOManager(
            loger=self.loger,
            coco_path=self._test_dataset_coco,
            name_dataset=self._test_dataset_name,
        )

    def run(self, grid):
        results = []
        for name, model_params in grid.items():
            result = self._run_one(name, model_params)
            result["name"] = name
            results.append(result)
        self._save_results(results)

    def _run_one(self, name, model_params):
        datasets = self._load_datasets(name, model_params)
        model_info = self._build_model(name, model_params)
        self._train(model_info, datasets["train"], model_params)
        result = self._test(name, model_info, datasets, model_params)
        return result

    def _load_datasets(self, name, model_params):
        tokenizer = self._get_tokenizer(name, model_params)
        pred_train = PredProcessor(loger=self.loger, coco_manager=self._coco_manager_train, tokenizer=tokenizer)
        pred_test = PredProcessor(loger=self.loger, coco_manager=self._coco_manager_test, tokenizer=tokenizer)

        cache_dir = model_params.get("_cache_dir", self._cache_pdf)
        train_dataset = GLAMDataset(
            coco_manager=self._coco_manager_train,
            default_index=0,
            pred=pred_train,
            loger=self.loger,
            cache_dir=cache_dir,
            pdf_dir=self._train_dataset_path,
        )
        N = len(train_dataset)
        for i, _ in enumerate(train_dataset):
            print(f"{(i + 1) / N * 100:4.2f} %", end="\r")

        test_dataset = GLAMDataset(
            coco_manager=self._coco_manager_test,
            default_index=0,
            pred=pred_test,
            loger=self.loger,
            cache_dir=cache_dir,
            pdf_dir=self._test_dataset_path,
        )
        N = len(test_dataset)
        for i, _ in enumerate(test_dataset):
            print(f"{(i + 1) / N * 100:4.2f} %", end="\r")

        return {"train": train_dataset, "test": test_dataset}

    def _build_model(self, name, model_params):
        if self._model_factory:
            return self._model_factory(name, model_params)

        params = {k: v for k, v in model_params.items() if not k.startswith("_")}
        model_name = str(Path(self.result_path, f"row2region_GLAM_{name}"))
        return {"model_name": model_name, "model_params": params}

    def _train(self, model_info, train_dataset, model_params):
        model_name = model_info["model_name"]
        params = copy.deepcopy(model_info["model_params"])

        train_dataset.train()

        node_imbalance, edge_imbalance = calculate_imbalance(train_dataset)
        num_classes = len(self._coco_manager_train.classes)
        if any(math.isnan(v) for v in node_imbalance):
            node_imbalance = [1.0] * num_classes
        if math.isnan(edge_imbalance):
            edge_imbalance = 1.0
        params.setdefault("loss_params", {})["node_imbalance"] = node_imbalance
        params.setdefault("loss_params", {})["edge_imbalance"] = edge_imbalance

        if Path(model_name).exists():
            return

        params["node_classifier_block"]["linear_post"][-1]["activation"] = "none"
        params["edge_classifier_block"]["linear_post"][-1]["activation"] = "none"

        model, num_restart = get_model(params, model_name)
        params["restart_num"] = num_restart
        loss = get_loss(params["loss_params"])
        trainer = Trainer(model=model, dataset=train_dataset, loss=loss, train_param=params, loger=self.loger)
        trainer.start_train()
        model = trainer.model
        save_model(model, model_name)

    def _test(self, name, model_info, datasets, model_params):
        if self._tester_factory:
            return self._tester_factory(name, model_info, datasets, model_params)

        test_dataset = datasets["test"]
        train_dataset = datasets["train"]
        model_name = model_info["model_name"]
        tokenizer = self._get_tokenizer(name, model_params)

        tmp_rez = Path(f"{model_name}_res.txt")
        if tmp_rez.exists():
            with open(tmp_rez, "r") as f:
                return json.load(f)

        params = copy.deepcopy(model_info["model_params"])
        params["node_classifier_block"]["linear_post"][-1]["activation"] = "softmax"
        params["edge_classifier_block"]["linear_post"][-1]["activation"] = "sigmoid"

        model, _ = get_model(params, model_name)

        model_id2name = train_dataset.coco_manager.classes
        dataset_name2id = {val: key for key, val in model_id2name.items()}
        pred = PredProcessor(loger=self.loger, tokenizer=tokenizer)
        pipeline = Pipeline(pred=pred, model=model, dataset_name2id=dataset_name2id, model_id2name=model_id2name)
        tester = Tester(pipeline=pipeline, dataset=test_dataset, loger=self.loger)
        tester.calculate()
        grid_cls, map_cls = tester.get_results()
        rez = {**grid_cls, **map_cls}
        with open(tmp_rez, "w") as f:
            json.dump(rez, f)
        return rez

    def _save_results(self, results):
        import pandas as pd

        df = pd.DataFrame(results)
        df.to_csv(self.result_path / "results.csv")
