import sys
import os
import torch
import numpy as np
from pager import ImageSegment
from rows2regionsGLAM.metrics import MultiGridMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from rows2regionsGLAM.utils.tester.metrics_per_class import classification_metrics_iou, classification_metrics_grid

DOC2PUB_MAP = {
    'Text': 'text',
    'Title': 'header',
    'Section-header': 'header',
    'List-item': 'list',
    'Table': 'table',
    'Picture': 'figure',
    'Page-header': 'other',
    'Page-footer': 'other',
    'Caption': 'other',
    'Footnote': 'other',
    'Formula': 'other',
    'other': 'other'
}


class Tester:
    def __init__(self, **conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']

        if "pred" not in conf.keys():
            raise Exception('Создайте и передайте predProcessor')
        else:
            self.pred = conf['pred']
        if "rows_model" not in conf.keys():
            raise Exception('Создайте и передайте rows_model')
        else:
            self.rows_model = conf['rows_model']
        if "region_model" not in conf.keys():
            raise Exception('Создайте и передайте region_model')
        else:
            self.region_model = conf['region_model']
        if "rows2regions" not in conf.keys():
            raise Exception('Создайте и передайте rows2regions')
        else:
            self.rows2regions = conf['rows2regions']
        self.loger.time_log()
        self.loger("Create Tester")

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

    def clean_true_regions(self, true_regions):
        return [reg for reg in true_regions if reg['segment']['height'] > 3 and reg['segment']['width'] > 3]



    def calculate_target_and_preds(self, test_dataset):
        target = []
        preds = []
        word_grids = []
        row_grids = []
        target_cls = []
        preds_cls = []

        id2name = test_dataset.coco_manager.coco_classes
        name2id = {name:id_ for id_, name in id2name.items()}
        
        N = len(test_dataset)
        for i, d in enumerate(test_dataset):
            name_file = test_dataset.pdf_names[i]
            true_regions = test_dataset.coco_manager.regions[name_file]['regions']
            clean_bboxes = self.clean_true_regions(true_regions)
            bboxes_true =[reg['segment'] for reg in clean_bboxes]
            classes_true = [reg['category_id'] for reg in clean_bboxes]
            row_json, pdf_img = self.pred.get_json_and_img(test_dataset.pdf_dir/name_file)
            row_json = row_json['rows']
            self.rows_model.from_dict({"rows": row_json})
           
            try:
                self.rows2regions.convert(self.rows_model, self.region_model, pdf_img)
                pred_regions = self.region_model.to_dict()['regions']

                # if name_dataset == "doclaynet" and name_test_dataset == "publaynet":
                #     for r in pred_regions:
                #         pub_label = DOC2PUB_MAP.get(r['label'], 'other')
                #         r['label'] = pub_label
                #     # pred_regions = aggregate_list_items(pred_regions)

                filtered_preds = [r for r in pred_regions if r['label'] != 'other']

                bboxes_pred = [r['segment'] for r in filtered_preds]
                classes_pred = [name2id[r['label']] for r in filtered_preds]

                # Очистка строк только для тестирования, в момент работы модели используются все строки, поскольку она училась на всех.
                # self.clean_rows(row_json, bboxes_true)
                word_grids.append([self.get_bbox(word['segment']) for row in row_json for word in row['words']])
                row_grids.append([self.get_bbox(row['segment']) for row in row_json])
                target.append([self.get_bbox(seg) for seg in bboxes_true])
                preds.append([self.get_bbox(seg) for seg in bboxes_pred])
                target_cls.append(classes_true)
                preds_cls.append(classes_pred)
            except Exception as e:
                print(e)
                print(i,d["file_name"])

            print(f"{(i + 1) / N * 100:4.2f} %", end='\r')

        return [target, preds, word_grids, row_grids, target_cls, preds_cls, id2name]

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


    def get_results(self, metrics):
        target, preds, word_grids, row_grids, target_cls, preds_cls, id2name = metrics
        map_cls = self.calculate_map_with_classes(preds, target, preds_cls, target_cls, dict_classes=id2name)
        grid_cls = self.calculate_grid_metric(preds, target, preds_cls, target_cls, word_grids, row_grids, dict_classes=id2name)
        return  grid_cls, map_cls
