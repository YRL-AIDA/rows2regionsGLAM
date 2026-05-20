import sys
import os
import torch
import numpy as np
from pager import ImageSegment
from rows2regionsGLAM.metrics import MultiGridMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from rows2regionsGLAM.utils.tester.metrics_per_class import classification_metrics_iou, classification_metrics_grid

class Tester:
    def __init__(self, **conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']

        if "pipeline" not in conf.keys():
            raise Exception('Создайте и передайте "Pipeline"')
        else:
            self.pipeline = conf['pipeline']
            
        if "dataset" not in conf.keys():
            raise Exception('Создайте и передайте rows_model')
        else:
            self.dataset = conf['dataset']
            
        self.loger.time_log()
        self.loger("Create Tester")
        self.data = None

    def get_bbox(self, segment, resize=None, delta_w=0, delta_h=0):
        coef_w, coef_h = 1, 1
        if resize:
            coef_w, coef_h = resize
        if not "height" in segment:
            segment['width'] = segment['x_bottom_right'] - segment['x_top_left']
            segment['height'] = segment['y_bottom_right'] - segment['y_top_left']
        return [
            int(coef_w * segment['x_top_left'] - delta_w),
            int(coef_h * segment['y_top_left'] - delta_h),
            int(coef_w * segment['width'] + delta_w),
            int(coef_h * segment['height'] + delta_h)
        ]

    def clean_rows(self, rows, bboxes_true):
        def is_row_in_region(row, segs_regions):
            for r in segs_regions:
                if row.is_intersection(r):
                    return True
            return False

        def is_good_block(block):
            h = block['segment']['y_bottom_right'] - block['segment']['y_top_left']
            w = block['segment']['x_bottom_right'] - block['segment']['x_top_left']
            return h > 3 and w > 3

        old_rows = [row for row in rows if is_good_block(row)]
        segs_row = [ImageSegment(dict_2p=row['segment']) for row in old_rows]
        new_rows = []
        seg_bboxes_true = [ImageSegment(dict_p_size=bbox) for bbox in bboxes_true]
        for i, row in enumerate(segs_row):
            if is_row_in_region(row, seg_bboxes_true):
                new_rows.append(old_rows[i])
        rows.clear()
        rows.extend(new_rows)

   



    def calculate(self):
        target = []
        preds = []
        word_grids = []
        row_grids = []
        target_cls = []
        preds_cls = []

        id2name = self.dataset.coco_manager.coco_classes
        self.dataset.test()
        N = len(self.dataset)
        for i, d in enumerate(self.dataset):
            try:
                bboxes_true  = d['bboxes_true']
                classes_true = d['classes_true']
                json_page = self.pipeline(d['path'])
                row_json = [row for reg in json_page['regions'] for row in reg['rows']]
                bboxes_pred =  [reg['segment'] for reg in json_page['regions']]
                classes_pred = [reg['label'] for reg in json_page['regions']]
                word_grids_ = [self.get_bbox(word['segment']) for row in row_json for word in row['words']]
                row_grids_ = [self.get_bbox(row['segment']) for row in row_json]
                target_ = [self.get_bbox(seg) for seg in bboxes_true]
                preds_ = [self.get_bbox(seg) for seg in bboxes_pred]
                
                word_grids.append(word_grids_)
                row_grids.append(row_grids_)
                target.append(target_)
                preds.append(preds_)
                target_cls.append(classes_true)
                preds_cls.append(classes_pred)
                
            except Exception as e:
                print(e)
                # print(i,d["file_name"])

            print(f"{(i + 1) / N * 100:4.2f} %", end='\r')

        self.data = [target, preds, word_grids, row_grids, target_cls, preds_cls, id2name]

    def calculate_grid_metric(self, preds, target, preds_cls, target_cls, word_grids, row_grids, dict_classes=None):
        grid_metric = MultiGridMetric()
        grid_metric.update(
            [dict(
                boxes=b,
                labels=cls
            ) for b, cls in zip(preds, preds_cls)],

            [dict(
                boxes=b,
                labels=cls
            ) for b, cls in zip(target, target_cls)], 
            word_grids, row_grids
        )
        rez = grid_metric.compute()
        return rez



    def calculate_map_with_classes(self, preds, target, preds_cls, target_cls, dict_classes=None):
        map_metric = MeanAveragePrecision(box_format="xywh", class_metrics=True)
        map_metric_seg = MeanAveragePrecision(box_format="xywh")

        map_metric.update(
            [dict(
                boxes=torch.tensor(b),
                scores=torch.tensor([1.0] * len(b)),
                labels=torch.tensor(cls)
            ) for b, cls in zip(preds, preds_cls)],

            [dict(
                boxes=torch.tensor(b),
                labels=torch.tensor(cls)
            ) for b, cls in zip(target, target_cls)]
        )
        rez = map_metric.compute()

        get_category = lambda an: 1
        map_metric_seg.update([dict(
            boxes=torch.tensor(bboxes_pred),
            scores=torch.tensor([1.0 for an in bboxes_pred]),
            labels=torch.tensor([get_category(an) for an in bboxes_pred]),
        ) for bboxes_pred in preds],
            [dict(
                boxes=torch.tensor(bboxes_true),
                labels=torch.tensor([get_category(an) for an in bboxes_true]),
            ) for bboxes_true in target])
        rez_seg = map_metric_seg.compute()
        
        reg_classes = rez['classes']
        reg_per = rez['map_per_class']
        dict_rez = {
            "name": "mAP@IoU[0.50:0.95]",
            "mAP (all)": float(rez['map']),
            "mAP (seg)": float(rez_seg['map'])
        }
        for map_cls, cls in zip(reg_per, reg_classes):
            dict_rez[f"mAP ({int(cls)if dict_classes is None else dict_classes[int(cls)]})"] = float(map_cls)
        return dict_rez 

    def print_result(self, metrics):
        grid_cls, map_cls = self.get_results(metrics)
        seg_str_map =f'{map_cls["name"]} (segmentation): {map_cls['segmentation']:.4f}'
        cls_str_map =f'{map_cls["name"]} (with classification): {map_cls['all']:.4f}'
      
        print(seg_str_map)
        print(cls_str_map)
        print(grid_cls)
        self.loger("Test Result")
        self.loger(seg_str_map)
        self.loger(cls_str_map)
        self.loger(grid_cls)


    def get_results(self):
        target, preds, word_grids, row_grids, target_cls, preds_cls, id2name = self.data
        map_cls = self.calculate_map_with_classes(preds, target, preds_cls, target_cls, dict_classes=id2name)
        grid_cls = self.calculate_grid_metric(preds, target, preds_cls, target_cls, word_grids, row_grids, dict_classes=id2name)
        return  grid_cls, map_cls
