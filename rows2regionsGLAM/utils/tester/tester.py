import sys
import os
import torch
import numpy as np
from pager import ImageSegment
from rows2regionsGLAM.metrics import GridMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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
    def __init__(self, conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        if "pdf_manager" not in conf.keys():
            raise Exception('Создайте и передайте pdf_manager')
        else:
            self.pdf_manager = conf['pdf_manager']
        if "row_manager" not in conf.keys():
            raise Exception('Создайте и передайте row_manager')
        else:
            self.row_manager = conf['row_manager'] 
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

    def get_bbox(self, segment, resize = None, delta_w = 0, delta_h = 0):
        coef_w, coef_h = 1, 1
        if resize:
            coef_w, coef_h = resize
        if not "height" in segment:
            segment['width'] = segment['x_bottom_right']-segment['x_top_left']
            segment['height']= segment['y_bottom_right']-segment['y_top_left']
        return [
            int(coef_w*segment['x_top_left']-delta_w),
            int(coef_h*segment['y_top_left']-delta_h),
            int(coef_w*segment['width']+delta_w),
            int(coef_h*segment['height']+delta_h)
        ]

    def clean_rows(self, rows, bboxes_true):
        def is_row_in_region(row, segs_regions):
            for r in segs_regions:
                if row.is_intersection(r):
                    return True
            return False
        def is_good_block(block):
            h = block['segment']['y_bottom_right']-block['segment']['y_top_left']
            w = block['segment']['x_bottom_right']-block['segment']['x_top_left']
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

    def clean_bboxes_true(self, bboxes_true):
        return [bbox_true for bbox_true in bboxes_true if bbox_true['height'] > 3 and bbox_true['width'] > 3]


    def calculate_target_and_preds(self, test_dataset, name_dataset, name_test_dataset, dataset_path, test_path):
        target = []
        preds = []
        word_grids = []
        row_grids = []
        N = len(test_dataset)
        for i, d in enumerate(test_dataset):
            name_file = test_dataset.pdf_names[i]
            true_regions = test_dataset.coco_ann[name_file]['regions']
            bboxes_true = self.clean_bboxes_true([reg['segment'] for reg in true_regions])
            
            pdf_json, pdf_img = self.pdf_manager.get_json_and_img_from_pdf(os.path.join(test_path, name_file))
            w, h = pdf_json['width'],pdf_json['height']
            if name_test_dataset == "doclaynet":
                resize = (w/1024, h/1024)
            else:
                resize = (1, 1)
            row_json = self.row_manager.get_row_json_from_pdf_json(pdf_json)   
            
            self.rows_model.from_dict({"rows": row_json})
            try:
                self.rows2regions.convert(self.rows_model, self.region_model, pdf_img)
                pred_regions = self.region_model.to_dict()['regions']

                if name_dataset == "doclaynet" and name_test_dataset == "publaynet":
                    for r in pred_regions:
                        pub_label = DOC2PUB_MAP.get(r['label'], 'other')
                        r['label'] = pub_label
                    # pred_regions = aggregate_list_items(pred_regions)

                bboxes_pred = [r['segment'] for r in pred_regions if r['label'] != 'other']

                # Очистка строк только для тестирования, в момент работы модели используются все строки, поскольку она училась на всех.
                # self.clean_rows(row_json, bboxes_true)
                word_grids.append([self.get_bbox(word['segment']) for row in row_json for word in row['words']])
                row_grids.append([self.get_bbox(row['segment']) for row in row_json])
                target.append([self.get_bbox(seg, resize) for seg in bboxes_true])
                preds.append([self.get_bbox(seg) for seg in bboxes_pred])
            except:
                print(i,d["file_name"])

            
            print(f"{(i+1)/N*100:4.2f} %", end='\r')

        return [target, preds, word_grids, row_grids]

    def calculate_grid_metric(self,  target, preds, word_grids, row_grids):
        grid_metric = GridMetric()

        i = 0
        N = len(preds)
        for bboxes_pred, bboxes_true, grid_row, grid_word in zip(preds, target, row_grids, word_grids):
            i += 1
            try:
                grid_metric.add_pair(bboxes_pred, bboxes_true, grid_row, grid_word)
                print(f"{(i) / N * 100:4.2f} %", end='\r')
            except:
                pass
        grid_metric.update()

        return grid_metric

    def calculate_map_metric(self, preds, target):
        map_metric = MeanAveragePrecision(box_format="xywh")

        get_category = lambda an: 1

        map_metric.update([dict(
            boxes=torch.tensor(bboxes_pred),
            scores=torch.tensor([1.0 for an in bboxes_pred]),
            labels=torch.tensor([get_category(an) for an in bboxes_pred]),
        ) for bboxes_pred in preds],
            [dict(
                boxes=torch.tensor(bboxes_true),
                labels=torch.tensor([get_category(an) for an in bboxes_true]),
            ) for bboxes_true in target])
        rez = map_metric.compute()
        map_metric_rez = f"mAP@IoU[0.50:0.95]   :{rez['map']:.8f}"

        return map_metric_rez

    def print_result(self, metrics):
        target = metrics[0]
        preds = metrics[1]
        word_grids = metrics[2]
        row_grids = metrics[3]

        map_metric_rez = self.calculate_map_metric(preds, target)
        grid_metric = self.calculate_grid_metric(target, preds, word_grids, row_grids)

        print(map_metric_rez)
        print(grid_metric)
        self.loger("Test Result")
        self.loger(map_metric_rez)
        self.loger(grid_metric.__str__())

    def get_results(self, metrics):
        target = metrics[0]
        preds = metrics[1]
        word_grids = metrics[2]
        row_grids = metrics[3]

        map_metric_rez = self.calculate_map_metric(preds, target)
        grid_metric = self.calculate_grid_metric(target, preds, word_grids, row_grids)

        return map_metric_rez, grid_metric.rez
