import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.core import init, ExperimentRunner
from experiments.core.model_params import default_arch
from rows2regionsGLAM.pipeline import Pipeline
from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.utils.tester import Tester
from rows2regionsGLAM.models import get_model


def _compute_iou(box1: list, box2: list) -> float:
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]
    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    intersection = inter_w * inter_h
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


class AnalysisErrorRunner(ExperimentRunner):
    def _run_one(self, name: str, model_params: dict):
        datasets = self._get_or_load_datasets(name, model_params)
        if datasets["train"] is not None:
            model_info = self._build_model(name, model_params)
            if "model_params" in model_info:
                self._train(model_info, datasets["train"], model_params,
                            val_dataset=datasets.get("val"))
        else:
            model_info = self._build_model(name, model_params)
        result = self._test(name, model_info, datasets, model_params)
        result["name"] = name
        return result

    def _test(self, name: str, model_info: dict, datasets: dict,
              model_params: dict) -> dict:
        test_dataset = datasets["test"]
        train_dataset = datasets["train"]
        model_name = model_info["model_name"]
        tokenizer = self._get_tokenizer(name, model_params)

        params = copy.deepcopy(model_info["model_params"])
        params["node_classifier_block"]["linear_post"][-1]["activation"] = "softmax"
        params["edge_classifier_block"]["linear_post"][-1]["activation"] = "sigmoid"

        model, _ = get_model(params, model_name)

        model_id2name = train_dataset.coco_manager.classes
        dataset_name2id = {val: key for key, val in model_id2name.items()}
        pred = PredProcessor(loger=self.loger, tokenizer=tokenizer)
        pipeline = Pipeline(pred=pred, model=model,
                            dataset_name2id=dataset_name2id,
                            model_id2name=model_id2name)
        tester = Tester(pipeline=pipeline, dataset=test_dataset, loger=self.loger)
        tester.calculate()
        grid_cls, map_cls = tester.get_results()
        rez = {**grid_cls, **map_cls}

        tmp_rez = Path(f"{model_name}_res.txt")
        with open(tmp_rez, "w") as f:
            json.dump(rez, f)

        target_bboxes, preds_bboxes, _, _, target_cls, preds_cls, id2name = tester.data
        num_classes = len(id2name)
        class_names = [id2name[i] for i in range(num_classes)]

        confusion = np.zeros((num_classes, num_classes), dtype=int)

        for doc_idx in range(len(target_bboxes)):
            gt_boxes = target_bboxes[doc_idx]
            gt_labels = target_cls[doc_idx]
            pd_boxes = preds_bboxes[doc_idx]
            pd_labels = preds_cls[doc_idx]
            if len(pd_boxes) == 0:
                continue
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                best_iou = 0.0
                best_pd_label = gt_label
                for pd_box, pd_label in zip(pd_boxes, pd_labels):
                    iou = _compute_iou(gt_box, pd_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_pd_label = pd_label
                confusion[gt_label, best_pd_label] += 1

        confusion_json = Path(self.result_path, f"{name}_confusion.json")
        confusion_dict = {
            "classes": class_names,
            "confusion_matrix": confusion.tolist(),
        }
        with open(confusion_json, "w") as f:
            json.dump(confusion_dict, f, indent=2)

        confusion_csv = Path(self.result_path, f"{name}_confusion.csv")
        df_cm = pd.DataFrame(confusion, index=class_names, columns=class_names)
        df_cm.index.name = "true"
        df_cm.columns.name = "pred"
        df_cm.to_csv(confusion_csv)

        self.loger(f"Confusion matrix saved: {confusion_csv}")

        print(f"\n{'=' * 60}")
        print(f"Confusion Matrix — {name}")
        print(f"{'=' * 60}")
        print(df_cm.to_string())
        print(f"{'=' * 60}\n")

        return rez


if __name__ == '__main__':
    init()
    EPOCHS = int(os.environ.get("EPOCHS", "30"))

    runner = AnalysisErrorRunner("result")

    configs = {}
    for seed in range(3):
        name = f"base_seed_{seed}"
        arch = default_arch(
            epochs=EPOCHS,
            early_stopping_patience=3,
            seed=seed,
        )
        configs[name] = arch

    runner.run(configs)