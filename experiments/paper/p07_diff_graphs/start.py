import json
import multiprocessing
import os
import signal
import time
from multiprocessing import cpu_count
from pathlib import Path

from experiments.core import init, ExperimentRunner
from experiments.paper.p07_diff_graphs.pipeline import (
    TrueModel,
    get_tokenizer as get_dg_tokenizer,
    DiffGraphsPipeline,
)
from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.utils.tester import Tester
from rows2regionsGLAM.utils.tester.tester import get_bbox

init()


_worker_coco_manager = None
_worker_pipeline = None
_worker_pdf_dir = None

ORACLE_TIMEOUT = 120


def _timeout_handler(signum, frame):
    raise TimeoutError("oracle timeout")


def _init_parallel_worker(pdf_dir, coco_path, name_dataset, name_tok,
                           default_index, num_classes, model_id2name, dataset_name2id):
    global _worker_coco_manager, _worker_pipeline, _worker_pdf_dir
    signal.signal(signal.SIGALRM, _timeout_handler)
    pid = os.getpid()
    time.sleep((pid % 10) * 0.3)
    from rows2regionsGLAM.utils.coco_manager import COCOManager
    from rows2regionsGLAM.pred_processor import PredProcessor
    from experiments.paper.p07_diff_graphs.pipeline import (
        TrueModel, get_tokenizer as _get_dg_tok, DiffGraphsPipeline as _DGP,
    )

    _worker_coco_manager = COCOManager(loger=None, coco_path=coco_path, name_dataset=name_dataset)
    tokenizer = _get_dg_tok(name_tok)
    pred = PredProcessor(loger=None, tokenizer=tokenizer)
    model = TrueModel()
    _worker_pipeline = _DGP(
        pred=pred, model=model,
        dataset_name2id=dataset_name2id,
        model_id2name=model_id2name,
        coco_manager=_worker_coco_manager,
        default_index=default_index,
        num_classes=num_classes,
    )
    _worker_pdf_dir = pdf_dir


def _oracle_one(pdf_name):
    signal.alarm(ORACLE_TIMEOUT)
    try:
        pdf_path = Path(_worker_pdf_dir) / pdf_name
        result = _worker_pipeline(pdf_path)
        true_regions, classes_true = _worker_coco_manager(pdf_name, {"width": 1, "height": 1})
        clean_bboxes, clean_classes = _worker_coco_clean(true_regions, classes_true)
        bboxes_true = [r.get_segment_p_size() for r in clean_bboxes]
        bboxes_pred = [reg['segment'] for reg in result['regions']]
        classes_pred = [reg['label'] for reg in result['regions']]
        word_grids = []
        row_grids = []
        for reg in result['regions']:
            for row in reg['rows']:
                row_grids.append(row['segment'])
                for word in row['words']:
                    word_grids.append(word['segment'])
        return (bboxes_true, clean_classes, bboxes_pred, classes_pred, word_grids, row_grids, pdf_name)
    except TimeoutError:
        print(f"SKIP (timeout >{ORACLE_TIMEOUT}s): {pdf_name}")
        return None
    finally:
        signal.alarm(0)


def _worker_coco_clean(true_regions, true_classes):
    clean_bboxes = []
    classes_true = []
    for cl, reg in zip(true_classes, true_regions):
        if reg.height > 3 and reg.width > 3:
            clean_bboxes.append(reg)
            classes_true.append(cl)
    return clean_bboxes, classes_true


def _calculate_oracle_parallel(test_dataset, model_params):
    name_tok = model_params["name_tok"]
    dataset_name2id = {val: key for key, val in test_dataset.coco_manager.classes.items()}
    model_id2name = test_dataset.coco_manager.classes

    total = len(test_dataset)
    workers = min(cpu_count(), 4)
    pdf_dir = str(test_dataset.pdf_dir)
    coco_path = test_dataset.coco_manager.coco_path
    name_dataset = test_dataset.coco_manager.name_dataset
    default_index = test_dataset.default_index
    num_classes = test_dataset.count_class

    ctx = multiprocessing.get_context('spawn')
    target = []
    preds = []
    word_grids = []
    row_grids = []
    target_cls = []
    preds_cls = []

    try:
        from tqdm import tqdm
        with ctx.Pool(
            workers,
            initializer=_init_parallel_worker,
            initargs=(pdf_dir, coco_path, name_dataset, name_tok,
                       default_index, num_classes, model_id2name, dataset_name2id),
        ) as pool:
            for item in tqdm(pool.imap_unordered(_oracle_one, test_dataset.pdf_names), total=total):
                if item is None:
                    continue
                bboxes_true, classes_true, bboxes_pred, classes_pred, w_grids, r_grids, pdf_name = item
                target_ = [get_bbox(seg) for seg in bboxes_true]
                preds_ = [get_bbox(seg) for seg in bboxes_pred]
                word_grids_ = [get_bbox(w) for w in w_grids]
                row_grids_ = [get_bbox(r) for r in r_grids]
                target.append(target_)
                preds.append(preds_)
                target_cls.append(classes_true)
                preds_cls.append(classes_pred)
                word_grids.append(word_grids_)
                row_grids.append(row_grids_)
    except ImportError:
        with ctx.Pool(
            workers,
            initializer=_init_parallel_worker,
            initargs=(pdf_dir, coco_path, name_dataset, name_tok,
                       default_index, num_classes, model_id2name, dataset_name2id),
        ) as pool:
            for item in pool.imap_unordered(_oracle_one, test_dataset.pdf_names):
                if item is None:
                    continue
                bboxes_true, classes_true, bboxes_pred, classes_pred, w_grids, r_grids, pdf_name = item
                target_ = [get_bbox(seg) for seg in bboxes_true]
                preds_ = [get_bbox(seg) for seg in bboxes_pred]
                word_grids_ = [get_bbox(w) for w in w_grids]
                row_grids_ = [get_bbox(r) for r in r_grids]
                target.append(target_)
                preds.append(preds_)
                target_cls.append(classes_true)
                preds_cls.append(classes_pred)
                word_grids.append(word_grids_)
                row_grids.append(row_grids_)

    return target, preds, word_grids, row_grids, target_cls, preds_cls


def model_factory(name, params):
    return {
        "model_name": os.path.join("result", params["name_tok"]),
        "model": TrueModel(),
    }


def tester_factory(name, model_info, datasets, model_params):
    test_dataset = datasets["test"]
    model_name = model_info["model_name"]

    tmp_rez = Path(f"{model_name}_res.txt")
    if tmp_rez.exists():
        with open(tmp_rez, "r") as f:
            return json.load(f)

    id2name = test_dataset.coco_manager.coco_classes
    target, preds, word_grids, row_grids, target_cls, preds_cls = \
        _calculate_oracle_parallel(test_dataset, model_params)

    tester = Tester(pipeline=None, dataset=test_dataset, loger=runner.loger)
    tester.data = [target, preds, word_grids, row_grids, target_cls, preds_cls, id2name]
    grid_cls, map_cls = tester.get_results()
    rez = {**grid_cls, **map_cls}
    with open(tmp_rez, "w") as f:
        json.dump(rez, f)
    return rez


def tokenizer_factory(name, params):
    return get_dg_tokenizer(params.get("name_tok", "glam"))


if __name__ == '__main__':
    runner = ExperimentRunner(
        "result",
        model_factory=model_factory,
        get_tokenizer=tokenizer_factory,
        tester_factory=tester_factory,
    )
    runner.run({
        name_tok: {"name_tok": name_tok, "_cache_dir": f"tmp_diff_graphs_{name_tok}", "_test_only": True, "debug_render": False}
        for name_tok in ["glam", "all"]
    })