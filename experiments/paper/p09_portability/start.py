import json
import os
from pathlib import Path

from experiments.core import ExperimentRunner
from experiments.core.bootstrap import init
from experiments.core.model_params import default_arch
from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.pipeline import Pipeline
from rows2regionsGLAM.utils.tester import Tester
from rows2regionsGLAM.models import get_model
from rows2regionsGLAM.utils.coco_manager import COCOManager
from rows2regionsGLAM.datasetloaders.base_line_dataset import GLAMDataset
from rows2regionsGLAM.tokenizers import RowGLAMTokenizer

init()

EPOCHS = int(os.environ.get("EPOCHS", "30"))
ds_path = Path("/home/daniil/disk01_1TB/datasets/")

DOC_CACHE = ds_path / "tmp/cache_miner/"
PUB_CACHE = ds_path / "tmp/cache_miner_publaynet/"

DOC_TRAIN_PDF = ds_path / "DocLayNet_core_10k/PDF/"
DOC_TRAIN_COCO = ds_path / "DocLayNet_core_10k/train.json"
PUB_TRAIN_PDF = ds_path / "micro_publaynet_10k/pdfs/train/"
PUB_TRAIN_COCO = ds_path / "micro_publaynet_10k/publaynet/train.json"

DOC_TEST_PDF = ds_path / "DocLayNet_core_mini/PDF/"
DOC_TEST_COCO = ds_path / "DocLayNet_core_mini/train.json"
PUB_TEST_PDF = ds_path / "micro_publaynet_10k/pdfs/dev/"
PUB_TEST_COCO = ds_path / "micro_publaynet_10k/publaynet/val.json"

PUB_NAME2ID = {"other": 0, "text": 1, "title": 2, "list": 3, "table": 4, "figure": 5}
DOC_NAME2ID = {
    "other": 0, "Caption": 1, "Footnote": 2, "Formula": 3, "List-item": 4,
    "Page-footer": 5, "Page-header": 6, "Picture": 7, "Section-header": 8,
    "Table": 9, "Text": 10, "Title": 11,
}

DS_INFO = {
    "pub": (PUB_TRAIN_PDF, PUB_TRAIN_COCO, PUB_TEST_PDF, PUB_TEST_COCO, PUB_CACHE, 6),
    "doc": (DOC_TRAIN_PDF, DOC_TRAIN_COCO, DOC_TEST_PDF, DOC_TEST_COCO, DOC_CACHE, 12),
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


def make_dataset(ds, loger, tokenizer, is_test=False):
    train_pdf, train_coco, test_pdf, test_coco, cache, num_cls = DS_INFO[ds]
    pdf_dir = test_pdf if is_test else train_pdf
    coco_path = test_coco if is_test else train_coco
    cm = COCOManager(loger=loger, coco_path=coco_path, name_dataset=ds)
    pred = PredProcessor(loger=loger, tokenizer=tokenizer)
    dataset = GLAMDataset(
        coco_manager=cm, default_index=0, pred=pred,
        loger=loger, cache_dir=cache, pdf_dir=pdf_dir,
    )
    dataset.train()
    dataset.init()
    return dataset, cm


def model_factory(name, params):
    _, _, _, _, _, num_cls = DS_INFO[params["train_ds"]]
    model_params = default_arch(num_classes=num_cls, epochs=EPOCHS, batch_size=128, save_frequency=10)
    model_name = str(Path("result", f"row2region_GLAM_{params['train_ds']}"))
    return {"model_name": model_name, "model_params": model_params}


def tester_factory(name, model_info, datasets, model_params):
    train_ds = model_params["train_ds"]
    test_ds = model_params["test_ds"]
    _, cm_test = datasets["_cm_test"]
    test_dataset = datasets["test"]
    model_name = model_info["model_name"]

    dataset_name2id, model_id2name = MAPPINGS[(train_ds, test_ds)]

    tmp_rez = Path(f"{model_name}_on_{test_ds}_test.txt")
    if tmp_rez.exists():
        with open(tmp_rez, "r") as f:
            return json.load(f)

    params = model_info["model_params"].copy()
    params["node_classifier_block"]["linear_post"][-1]["activation"] = "softmax"
    params["edge_classifier_block"]["linear_post"][-1]["activation"] = "sigmoid"

    model, _ = get_model(params, model_name)
    tokenizer = RowGLAMTokenizer()
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
    def _load_datasets(self, name, model_params):
        train_ds = model_params["train_ds"]
        test_ds = model_params["test_ds"]
        tokenizer = RowGLAMTokenizer()
        train_dataset, cm_train = make_dataset(train_ds, self.loger, tokenizer)
        test_dataset, cm_test = make_dataset(test_ds, self.loger, tokenizer, is_test=True)
        return {"train": train_dataset, "test": test_dataset, "_cm_test": (cm_test,)}


runner = PortabilityRunner("result", model_factory=model_factory, tester_factory=tester_factory)
runner.run({
    f"train_{train_ds}_test_{test_ds}": {"train_ds": train_ds, "test_ds": test_ds}
    for train_ds in ["doc", "pub"]
    for test_ds in ["doc", "pub"]
})
