import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from experiments.core import ExperimentRunner
from experiments.core.bootstrap import init
from experiments.core.model_params import default_arch
from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.pipeline import Pipeline
from rows2regionsGLAM.utils.tester import Tester
from rows2regionsGLAM.models import get_model, get_loss, save_model
from rows2regionsGLAM.utils.coco_manager import COCOManager
from rows2regionsGLAM.datasetloaders.base_line_dataset import GLAMDataset
from rows2regionsGLAM.utils.trainer import Trainer
from rows2regionsGLAM.utils.imbalance import calculate_imbalance
from rows2regionsGLAM.tokenizers.font_emb_tokenizer import RowGLAMTokenizer as EmbFontTokenizer
from rows2regionsGLAM.tokenizers.font_tokenizer import RowGLAMTokenizer as PDFFontTokenizer
from rows2regionsGLAM.tokenizers import RowGLAMTokenizer as NoFontTokenizer

if __name__ == '__main__':
    init()
    EPOCHS = int(os.environ.get("EPOCHS", "30"))

    DOC_TRAIN_PDF = Path(os.environ["DOC_TRAIN_PATH"])
    DOC_TRAIN_COCO = Path(os.environ["DOC_TRAIN_COCO"])
    DOC_TEST_PDF = Path(os.environ["DOC_TEST_PATH"])
    DOC_TEST_COCO = Path(os.environ["DOC_TEST_COCO"])
    DOC_CACHE = Path(os.environ["DOC_CACHE"])

    PUB_TRAIN_PDF = Path(os.environ["PUB_TRAIN_PATH"])
    PUB_TRAIN_COCO = Path(os.environ["PUB_TRAIN_COCO"])
    PUB_TEST_PDF = Path(os.environ["PUB_TEST_PATH"])
    PUB_TEST_COCO = Path(os.environ["PUB_TEST_COCO"])
    PUB_CACHE = Path(os.environ["PUB_CACHE"])

    PUB_NAME2ID = {"other": 0, "text": 1, "title": 2, "list": 3, "table": 4, "figure": 5}
    DOC_NAME2ID = {
        "other": 0, "Caption": 1, "Footnote": 2, "Formula": 3, "List-item": 4,
        "Page-footer": 5, "Page-header": 6, "Picture": 7, "Section-header": 8,
        "Table": 9, "Text": 10, "Title": 11,
    }

    DS_INFO = {
        "pub": (PUB_TRAIN_PDF, PUB_TRAIN_COCO, PUB_TEST_PDF, PUB_TEST_COCO, PUB_CACHE, 6, "publaynet"),
        "doc": (DOC_TRAIN_PDF, DOC_TRAIN_COCO, DOC_TEST_PDF, DOC_TEST_COCO, DOC_CACHE, 12, "doclaynet"),
    }

    MAPPINGS = {
        ("pub", "pub"): (
            PUB_NAME2ID,
            {0: "other", 1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"},
        ),
        ("doc", "doc"): (
            DOC_NAME2ID,
            {0: "other", 1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item",
             5: "Page-footer", 6: "Page-header", 7: "Picture", 8: "Section-header",
             9: "Table", 10: "Text", 11: "Title"},
        ),
        ("doc", "pub"): (
            PUB_NAME2ID,
            {0: "other", 1: "text", 2: "text", 3: "other", 4: "list",
             5: "other", 6: "other", 7: "figure", 8: "title", 9: "table",
             10: "text", 11: "title"},
        ),
        ("pub", "doc"): (
            DOC_NAME2ID,
            {0: "other", 1: "Text", 2: "Section-header", 3: "List-item",
             4: "Table", 5: "Picture"},
        ),
    }

    TOKENIZER_INPUT_DIMS = {"font_emb": 527, "pdf_font": 18, "no_font": 15}

    emb_font_tokenizer = EmbFontTokenizer()
    pdf_font_tokenizer = PDFFontTokenizer()
    no_font_tokenizer = NoFontTokenizer()

    def get_tokenizer(name, params):
        tokenizer_name = params.get("_tokenizer", "no_font")
        if tokenizer_name == "font_emb":
            return emb_font_tokenizer
        elif tokenizer_name == "pdf_font":
            return pdf_font_tokenizer
        elif tokenizer_name == "no_font":
            return no_font_tokenizer
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

    def make_dataset(ds, loger, tokenizer, is_test=False):
        train_pdf, train_coco, test_pdf, test_coco, cache, num_cls, ds_name = DS_INFO[ds]
        pdf_dir = test_pdf if is_test else train_pdf
        coco_path = test_coco if is_test else train_coco
        cm = COCOManager(loger=loger, coco_path=coco_path, name_dataset=ds_name)
        pred = PredProcessor(loger=loger, tokenizer=tokenizer)
        dataset = GLAMDataset(
            coco_manager=cm, default_index=0, pred=pred,
            loger=loger, cache_dir=cache, pdf_dir=pdf_dir,
        )
        dataset.train()
        dataset.init()
        return dataset, cm

    def model_factory(name, params):
        train_ds = params["_train_ds"]
        tokenizer_name = params.get("_tokenizer", "no_font")
        input_dim = TOKENIZER_INPUT_DIMS[tokenizer_name]
        _, _, _, _, _, num_cls, _ = DS_INFO[train_ds]
        model_params = default_arch(
            input_dim=input_dim, num_classes=num_cls,
            epochs=EPOCHS, batch_size=128, early_stopping_patience=3,
        )
        model_name = str(Path("result", f"row2region_GLAM_{name}"))
        return {"model_name": model_name, "model_params": model_params}

    def tester_factory(name, model_info, datasets, model_params):
        train_ds = model_params["_train_ds"]
        test_ds = model_params["_test_ds"]
        cm_test = datasets["_cm_test"]
        test_dataset = datasets["test"]
        model_name = model_info["model_name"]
        tokenizer = get_tokenizer(name, model_params)

        dataset_name2id, model_id2name = MAPPINGS[(train_ds, test_ds)]

        tmp_rez = Path(f"{model_name}_on_{test_ds}_test.txt")
        if tmp_rez.exists():
            with open(tmp_rez, "r") as f:
                return json.load(f)

        params = model_info["model_params"].copy()
        params["node_classifier_block"]["linear_post"][-1]["activation"] = "softmax"
        params["edge_classifier_block"]["linear_post"][-1]["activation"] = "sigmoid"

        model, _ = get_model(params, model_name)
        pred = PredProcessor(loger=runner.loger, tokenizer=tokenizer)
        pipeline = Pipeline(pred=pred, model=model, dataset_name2id=dataset_name2id, model_id2name=model_id2name)
        tester = Tester(pipeline=pipeline, dataset=test_dataset, loger=runner.loger)
        tester.calculate()
        grid_cls, map_cls = tester.get_results()
        rez = {**grid_cls, **map_cls}
        with open(tmp_rez, "w") as f:
            json.dump(rez, f)
        return rez

    class PortabilityRunner(ExperimentRunner):
        def __init__(self, result_path, model_factory=None, get_tokenizer=None, tester_factory=None):
            from rows2regionsGLAM.utils.loger import Loger
            from rows2regionsGLAM.tokenizers import RowGLAMTokenizer

            self.result_path = Path(result_path)
            if not self.result_path.exists():
                self.result_path.mkdir(parents=True)
            self.loger = Loger()
            self._model_factory = model_factory
            self._tester_factory = tester_factory
            self._get_tokenizer = get_tokenizer or (lambda name, params: RowGLAMTokenizer())
            self._cache_pdf = None
            self._datasets_cache = {}

        def _load_datasets(self, name, model_params):
            train_ds = model_params["_train_ds"]
            test_ds = model_params["_test_ds"]
            tokenizer = self._get_tokenizer(name, model_params)
            train_dataset, cm_train = make_dataset(train_ds, self.loger, tokenizer)
            test_dataset, cm_test = make_dataset(test_ds, self.loger, tokenizer, is_test=True)
            return {"train": train_dataset, "test": test_dataset, "_cm_test": cm_test}

        def _dataset_key(self, name, model_params):
            train_ds = model_params["_train_ds"]
            test_ds = model_params["_test_ds"]
            tokenizer = self._get_tokenizer(name, model_params)
            train_cache = DS_INFO[train_ds][4]
            test_cache = DS_INFO[test_ds][4]
            return (train_cache, test_cache, type(tokenizer).__name__)

        def _get_or_load_datasets(self, name, model_params):
            key = self._dataset_key(name, model_params)
            if key not in self._datasets_cache:
                self._datasets_cache[key] = self._load_datasets(name, model_params)
            return self._datasets_cache[key]

        def _train(self, model_info, train_dataset, model_params):
            model_name = model_info["model_name"]
            params = copy.deepcopy(model_info["model_params"])

            train_dataset.train()

            node_imbalance, edge_imbalance = calculate_imbalance(train_dataset)
            num_classes = model_info["model_params"].get("NodeClasses", 6)
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
            trainer = Trainer(model=model, dataset=train_dataset, loss=loss, train_param=params, loger=self.loger)
            trainer.start_train()
            model = trainer.model
            save_model(model, model_name)

    runner = PortabilityRunner(
        "result",
        model_factory=model_factory,
        get_tokenizer=get_tokenizer,
        tester_factory=tester_factory,
    )
    runner.run({
        f"train_{train_ds}_test_{test_ds}_{tokenizer_name}": {
            "_train_ds": train_ds,
            "_test_ds": test_ds,
            "_tokenizer": tokenizer_name,
            **default_arch(
                input_dim=TOKENIZER_INPUT_DIMS[tokenizer_name],
                num_classes=DS_INFO[train_ds][5],
                epochs=EPOCHS,
                batch_size=128,
                early_stopping_patience=3,
            ),
        }
        for train_ds in ["doc", "pub"]
        for test_ds in ["doc", "pub"]
        for tokenizer_name in ["font_emb"]
    })
