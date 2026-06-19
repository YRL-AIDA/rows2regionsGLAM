import json
import os
from pathlib import Path

from experiments.core import init, ExperimentRunner
from experiments.paper.p07_diff_graphs.pipeline import (
    TrueModel,
    get_tokenizer as get_dg_tokenizer,
    DiffGraphsPipeline,
)
from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.utils.tester import Tester

init()


def model_factory(name, params):
    return {
        "model_name": os.path.join("result", params["name_tok"]),
        "model": TrueModel(),
    }


def tester_factory(name, model_info, datasets, model_params):
    test_dataset = datasets["test"]
    model_name = model_info["model_name"]
    tokenizer = get_dg_tokenizer(model_params["name_tok"])
    pred = PredProcessor(
        loger=runner.loger,
        coco_manager=runner._coco_manager_test,
        tokenizer=tokenizer,
    )

    tmp_rez = Path(f"{model_name}_res.txt")
    if tmp_rez.exists():
        with open(tmp_rez, "r") as f:
            return json.load(f)

    model = model_info["model"]
    model_id2name = test_dataset.coco_manager.classes
    dataset_name2id = {val: key for key, val in model_id2name.items()}

    debug_render_dir = None
    if model_params.get("debug_render"):
        debug_render_dir = Path(f"{model_name}_debug_pages")
        debug_render_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DiffGraphsPipeline(
        pred=pred,
        model=model,
        dataset_name2id=dataset_name2id,
        model_id2name=model_id2name,
        coco_manager=test_dataset.coco_manager,
        default_index=test_dataset.default_index,
        num_classes=test_dataset.count_class,
        debug_render_dir=str(debug_render_dir) if debug_render_dir else None,
    )
    tester = Tester(pipeline=pipeline, dataset=test_dataset, loger=runner.loger)
    tester.calculate()
    grid_cls, map_cls = tester.get_results()
    rez = {**grid_cls, **map_cls}
    with open(tmp_rez, "w") as f:
        json.dump(rez, f)
    return rez


def tokenizer_factory(name, params):
    return get_dg_tokenizer(params.get("name_tok", "glam"))


runner = ExperimentRunner(
    "result",
    model_factory=model_factory,
    get_tokenizer=tokenizer_factory,
    tester_factory=tester_factory,
)
runner.run({
    name_tok: {"name_tok": name_tok, "_cache_dir": f"tmp_diff_graphs_{name_tok}", "debug_render": False}
    for name_tok in ["glam", "all"]
})
