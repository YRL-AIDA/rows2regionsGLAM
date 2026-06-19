#!/usr/bin/env python3
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PATH_PROJECT = os.path.join('..', '..')
sys.path.append(PATH_PROJECT)
env_file = os.path.join(PATH_PROJECT, '.env')
load_dotenv(env_file)
warnings.filterwarnings('ignore')

from pager.page_model.sub_models import RegionModel, RowsModel
from pager.page_model.sub_models.dtype import ImageSegment

from utils.experimenter import Experimenter
from rows2regionsGLAM.utils.loger import Loger
from rows2regionsGLAM.utils.coco_manager import COCOManager
from rows2regionsGLAM.utils.cacher import Cacher
from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.datasetloaders.base_line_dataset import GLAMDataset
from rows2regionsGLAM.utils.trainer import Trainer
from rows2regionsGLAM.utils.imbalance import calculate_imbalance
from rows2regionsGLAM.utils.intersect_util import get_num_regions_of_rows
from rows2regionsGLAM.models import get_model, get_loss, save_model
from rows2regionsGLAM.tokenizers import RowGLAMTokenizer
from rows2regionsGLAM.pipeline import Pipeline
from rows2regionsGLAM.pipeline.converters.base_line_converter import Rows2Regions
from rows2regionsGLAM.metrics import MultiGridMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision

EXP_NAME = 'diff_graphs'
exp_path = os.path.join('experiments', EXP_NAME, 'result')
EPOCHS = int(os.environ.get('EPOCHS', '30'))

BASE_PARAMS = {
    'node_block': {
        'linear_pred': [
            {'in': 15, 'out': 256, 'activation': 'gelu'},
            {'in': 256, 'out': 128, 'activation': 'gelu'},
        ],
        'gnn': [
            ('tag', {'batch_norm': True, 'concat': True,
                     'in': 128, 'in_gnn': 256, 'out_gnn': 128,
                     'gnn_activation': 'gelu', 'activation': 'gelu', 'K': 3}),
            ('tag', {'batch_norm': False, 'concat': True,
                     'in': 128 + 128, 'in_gnn': 256, 'out_gnn': 128,
                     'gnn_activation': 'gelu', 'activation': 'gelu', 'K': 3}),
        ],
    },
    'node_classifier_block': {
        'linear_post': [
            {'in': 128 + 128 + 128, 'out': 256, 'activation': 'gelu'},
            {'in': 256, 'out': 128, 'activation': 'gelu'},
            {'in': 128, 'out': 6, 'activation': 'none'},
        ]
    },
    'post_node_block': {
        'linear_pred': [
            {'in': 128 + 128 + 128, 'out': 256, 'activation': 'gelu'},
            {'in': 256, 'out': 256, 'activation': 'gelu'},
            {'in': 256, 'out': 128, 'activation': 'gelu'},
        ],
        'gnn': [
            ('tag', {'batch_norm': True, 'concat': True,
                     'in': 128, 'in_gnn': 256, 'out_gnn': 128,
                     'gnn_activation': 'gelu', 'activation': 'gelu', 'K': 3}),
            ('tag', {'batch_norm': False, 'concat': True,
                     'in': 128 + 128, 'in_gnn': 256, 'out_gnn': 128,
                     'gnn_activation': 'gelu', 'activation': 'gelu', 'K': 3}),
        ],
        'linear_post': [
            {'in': 128 + 128 + 128, 'out': 256, 'activation': 'gelu'},
            {'in': 256, 'out': 128, 'activation': 'gelu'},
            {'in': 128, 'out': 128, 'activation': 'gelu'},
        ]
    },
    'edge_classifier_block': {
        'linear_post': [
            {'in': 2 * (128 + 15) + 4, 'out': 256, 'activation': 'gelu'},
            {'in': 256, 'out': 64, 'activation': 'gelu'},
            {'in': 64, 'out': 1, 'activation': 'none'},
        ]
    },
    'epochs': EPOCHS,
    'batch_size': 64,
    'learning_rate': 0.001,
    'seg_k': 0.5,
    'loss_params': {
        'edge_coef': 0.8,
        'node_coef': 0.2
    }
}


def _get_bbox(segment, resize=None, delta_w=0, delta_h=0):
    coef_w, coef_h = 1, 1
    if resize:
        coef_w, coef_h = resize
    if 'height' not in segment:
        segment['width'] = segment['x_bottom_right'] - segment['x_top_left']
        segment['height'] = segment['y_bottom_right'] - segment['y_top_left']
    return [
        int(coef_w * segment['x_top_left'] - delta_w),
        int(coef_h * segment['y_top_left'] - delta_h),
        int(coef_w * segment['width'] + delta_w),
        int(coef_h * segment['height'] + delta_h)
    ]


def _get_mini_seg(r):
    img_seg = ImageSegment(dict_2p=r)
    if img_seg.height < 5:
        return img_seg
    delta = int(img_seg.height / 5)
    img_seg.y_bottom_right = img_seg.y_bottom_right - delta
    img_seg.y_top_left = img_seg.y_top_left + delta
    return img_seg


def compute_gt_node_classes(rows, region_segs, region_categories):
    row_segments = [_get_mini_seg(row['segment']) for row in rows]
    nums_regions = get_num_regions_of_rows(region_segs, row_segments)

    gt_classes = []
    for row_seg, num_reg in zip(row_segments, nums_regions):
        if num_reg is None:
            gt_classes.append(None)
        else:
            gt_classes.append(region_categories[num_reg])
    return gt_classes


def gt_classes_to_onehot(gt_classes, num_classes, device):
    vecs = []
    for c in gt_classes:
        v = [0.0] * num_classes
        if c is not None:
            v[c] = 1.0
        else:
            v[0] = 1.0
        vecs.append(v)
    return torch.tensor(vecs, dtype=torch.float32, device=device)


def model_regions_from_graph(rows_json, graph, deleted_edges, node_classes_tensor, classes_dict):
    from pager.page_model.sub_models.dtype import Graph
    graph_ = Graph()
    for row_json in rows_json:
        segment = ImageSegment(dict_2p=row_json['segment'])
        xc, yc = segment.get_center()
        graph_.add_node(xc, yc)

    for node_i, node_j, ind in zip(graph[0], graph[1], deleted_edges):
        if not ind:
            graph_.add_edge(node_i + 1, node_j + 1)

    regions = []
    for reg in graph_.get_related_graphs():
        indexes = [node.index - 1 for node in reg.get_nodes()]
        row_classes = np.array([node_classes_tensor[i].detach().cpu().numpy() for i in indexes])
        label = classes_dict[np.argmax(row_classes.mean(axis=0) if row_classes.shape[0] > 1 else row_classes[0])]
        regions.append({'rows': [rows_json[i] for i in indexes], 'label': label})
    return regions


def fun_get_model_with_param(param):
    model_name = str(Path(exp_path, f'row2region_GLAM_{EXP_NAME}'))
    model_params = param.copy()
    return {
        'model_name': model_name,
        'model_params': model_params
    }


train_dataset_name = os.environ['NAME_DATASET']
train_dataset_path = os.environ['DATASET_PATH']
train_dataset_coco = os.environ['COCO_PATH']
test_dataset_name = os.environ['NAME_TEST_DATASET']
test_dataset_path = os.environ['TEST_PATH']
test_dataset_coco = os.environ['TEST_COCO_PATH']
cache_pdf = os.environ.get('CASH_PDF_PATH', 'tmp_cache')

loger = Loger(log_dir=os.path.join(exp_path, 'logs'))
coco_manager_train = COCOManager(loger=loger, coco_path=train_dataset_coco, name_dataset=train_dataset_name)
coco_manager_test = COCOManager(loger=loger, coco_path=test_dataset_coco, name_dataset=test_dataset_name)
tokenizer = RowGLAMTokenizer()
pred = PredProcessor(loger=loger, tokenizer=tokenizer)


def fun_get_dataset_with_param(param):
    train_dataset = GLAMDataset(
        coco_manager=coco_manager_train, default_index=0, pred=pred,
        loger=loger, cache_dir=cache_pdf, pdf_dir=train_dataset_path
    )
    train_dataset.train()
    train_dataset.init()

    test_dataset = GLAMDataset(
        coco_manager=coco_manager_test, default_index=0, pred=pred,
        loger=loger, cache_dir=cache_pdf, pdf_dir=test_dataset_path
    )
    test_dataset.train()
    test_dataset.init()
    return {'train': train_dataset, 'test': test_dataset}


def fun_train_model_with_param(model, dataset, param):
    model_name = model['model_name']
    model_params = model['model_params'].copy()
    model_params['node_classifier_block']['linear_post'][-1]['activation'] = 'none'
    model_params['edge_classifier_block']['linear_post'][-1]['activation'] = 'none'

    if Path(model_name).exists():
        return

    dataset_train = dataset['train']
    dataset_train.train()

    node_imbalance, edge_imbalance = calculate_imbalance(dataset_train)
    model_params['loss_params']['node_imbalance'] = node_imbalance
    model_params['loss_params']['edge_imbalance'] = edge_imbalance

    torch_model, _ = get_model(model_params, model_name)
    loss = get_loss(model_params['loss_params'])
    trainer = Trainer(model=torch_model, dataset=dataset_train, loss=loss, train_param=model_params, loger=loger)
    trainer.start_train()
    save_model(trainer.model, model_name)


def fun_test_model_with_param(model, dataset, param):
    model_name = model['model_name']
    model_params = model['model_params'].copy()
    model_params['node_classifier_block']['linear_post'][-1]['activation'] = 'softmax'
    model_params['edge_classifier_block']['linear_post'][-1]['activation'] = 'sigmoid'

    device = torch.device(os.environ.get('DEVICE', 'cpu'))
    model_id2name = coco_manager_train.classes
    model_id2name_inv = coco_manager_test.classes
    dataset_name2id = {val: key for key, val in model_id2name.items()}
    num_classes = len(model_id2name)

    torch_model, _ = get_model(model_params, model_name)
    torch_model.to(device)
    torch_model.eval()

    pdf_files = sorted([f for f in os.listdir(test_dataset_path) if f.endswith('.pdf')])

    target = []
    preds_gt = []
    preds_model = []
    word_grids = []
    row_grids = []
    target_cls = []
    preds_gt_cls = []
    preds_model_cls = []

    times_extract = []
    times_segment = []

    for i, pdf_name in enumerate(pdf_files):
        try:
            pdf_path = Path(test_dataset_path) / pdf_name

            t0 = time.perf_counter()
            rez = pred(pdf_path)
            t1 = time.perf_counter()
            times_extract.append(t1 - t0)

            rows = rez['pdf_json']['rows']
            torch_dict = rez['torch_dict']
            img = rez['img']

            true_regions, true_categories = coco_manager_test(pdf_name, rez['pdf_json'])
            clean_regions = []
            clean_categories = []
            for reg, cat in zip(true_regions, true_categories):
                if reg.height > 3 and reg.width > 3:
                    clean_regions.append(reg)
                    clean_categories.append(cat)

            bboxes_true = [reg.get_segment_p_size() for reg in clean_regions]
            target.append([_get_bbox(seg) for seg in bboxes_true])
            target_cls.append(clean_categories)

            gt_node_classes = compute_gt_node_classes(rows, clean_regions, clean_categories)

            X = torch.tensor(torch_dict['X'], dtype=torch.float32, device=device)
            Y = torch.tensor(torch_dict['Y'], dtype=torch.float32, device=device)
            N = torch_dict['N']
            inds_data = torch_dict['inds']
            index_for_mtrx = [inds_data[0] + inds_data[1], inds_data[1] + inds_data[0]]
            sp_A = torch.sparse_coo_tensor(
                indices=index_for_mtrx,
                values=[1.0] * len(index_for_mtrx[0]),
                size=(N, N), dtype=torch.float32, device=device
            )

            t2 = time.perf_counter()
            with torch.no_grad():
                result = torch_model({'X': X, 'Y': Y, 'sp_A': sp_A, 'inds': inds_data})

            node_classes_pred = result['node_classes']
            edge_pred = result['E_pred']
            deleted_edges = edge_pred < 0.5
            t3 = time.perf_counter()
            times_segment.append(t3 - t2)

            node_classes_gt = gt_classes_to_onehot(gt_node_classes, num_classes, device)

            regions_gt = model_regions_from_graph(
                rows, inds_data, deleted_edges, node_classes_gt, model_id2name_inv
            )
            regions_model = model_regions_from_graph(
                rows, inds_data, deleted_edges, node_classes_pred, model_id2name_inv
            )

            bboxes_gt = [reg['rows'][0] if len(reg['rows']) == 1 else {
                'x_top_left': min(r['segment']['x_top_left'] for r in reg['rows']),
                'y_top_left': min(r['segment']['y_top_left'] for r in reg['rows']),
                'x_bottom_right': max(r['segment']['x_bottom_right'] for r in reg['rows']),
                'y_bottom_right': max(r['segment']['y_bottom_right'] for r in reg['rows']),
            } for reg in regions_gt]

            if len(regions_gt) == 0:
                bboxes_gt_formatted = []
            elif isinstance(bboxes_gt[0], dict):
                bboxes_gt_formatted = [_get_bbox(seg) for seg in bboxes_gt]
            else:
                bboxes_gt_formatted = [_get_bbox(r['segment']) for r in bboxes_gt if 'segment' in r]

            bboxes_model = [reg['rows'][0] if len(reg['rows']) == 1 else {
                'x_top_left': min(r['segment']['x_top_left'] for r in reg['rows']),
                'y_top_left': min(r['segment']['y_top_left'] for r in reg['rows']),
                'x_bottom_right': max(r['segment']['x_bottom_right'] for r in reg['rows']),
                'y_bottom_right': max(r['segment']['y_bottom_right'] for r in reg['rows']),
            } for reg in regions_model]

            if len(regions_model) == 0:
                bboxes_model_formatted = []
            elif isinstance(bboxes_model[0], dict):
                bboxes_model_formatted = [_get_bbox(seg) for seg in bboxes_model]
            else:
                bboxes_model_formatted = [_get_bbox(r['segment']) for r in bboxes_model if 'segment' in r]

            classes_gt = [dataset_name2id.get(reg['label'], 0) for reg in regions_gt]
            classes_model = [dataset_name2id.get(reg['label'], 0) for reg in regions_model]

            preds_gt.append(bboxes_gt_formatted)
            preds_gt_cls.append(classes_gt)
            preds_model.append(bboxes_model_formatted)
            preds_model_cls.append(classes_model)

            word_grids_ = [_get_bbox(word['segment']) for row in rows for word in row.get('words', [])]
            row_grids_ = [_get_bbox(row['segment']) for row in rows]
            word_grids.append(word_grids_)
            row_grids.append(row_grids_)

        except Exception as e:
            print(f'Error on {pdf_name}: {e}')
            import traceback
            traceback.print_exc()

        print(f'{(i + 1) / len(pdf_files) * 100:4.2f} %', end='\r')

    print()

    map_gt = _compute_map(preds_gt, target, preds_gt_cls, target_cls, model_id2name)
    map_model = _compute_map(preds_model, target, preds_model_cls, target_cls, model_id2name)

    grid_gt = _compute_grid(preds_gt, target, preds_gt_cls, target_cls, word_grids, row_grids, model_id2name)
    grid_model = _compute_grid(preds_model, target, preds_model_cls, target_cls, word_grids, row_grids, model_id2name)

    total_extract = sum(times_extract)
    total_segment = sum(times_segment)
    n_docs = len(times_extract)

    result = {
        'name': EXP_NAME,
        'mode': 'diff_graphs_upper_bound',
        'n_docs': n_docs,
        'time_extract_total_s': round(total_extract, 2),
        'time_extract_mean_s': round(total_extract / n_docs, 4) if n_docs else 0,
        'time_segment_total_s': round(total_segment, 2),
        'time_segment_mean_s': round(total_segment / n_docs, 4) if n_docs else 0,
        'time_total_s': round(total_extract + total_segment, 2),
    }

    for k, v in map_gt.items():
        result[f'GT_{k}'] = v
    for k, v in map_model.items():
        result[f'MODEL_{k}'] = v
    for k, v in grid_gt.items():
        result[f'GRID_GT_{k}'] = v
    for k, v in grid_model.items():
        result[f'GRID_MODEL_{k}'] = v

    return result


def _compute_map(preds, target, preds_cls, target_cls, dict_classes):
    metric_all = MeanAveragePrecision(box_format='xywh', class_metrics=True)
    metric_seg = MeanAveragePrecision(box_format='xywh')

    metric_all.update(
        [dict(boxes=torch.tensor(b), scores=torch.tensor([1.0] * len(b)), labels=torch.tensor(cls))
         for b, cls in zip(preds, preds_cls)],
        [dict(boxes=torch.tensor(b), labels=torch.tensor(cls))
         for b, cls in zip(target, target_cls)]
    )
    rez = metric_all.compute()

    metric_seg.update(
        [dict(boxes=torch.tensor(b), scores=torch.tensor([1.0] * len(b)), labels=torch.tensor([1] * len(b)))
         for b in preds],
        [dict(boxes=torch.tensor(b), labels=torch.tensor([1] * len(b)))
         for b in target]
    )
    rez_seg = metric_seg.compute()

    result = {
        'mAP(all)': float(rez['map']),
        'mAP(seg)': float(rez_seg['map']),
    }
    for map_val, cls_id in zip(rez['map_per_class'], rez['classes']):
        cls_name = dict_classes.get(int(cls_id), str(int(cls_id)))
        result[f'mAP({cls_name})'] = float(map_val)
    return result


def _compute_grid(preds, target, preds_cls, target_cls, word_grids, row_grids, dict_classes):
    grid_metric = MultiGridMetric()
    grid_metric.update(
        [dict(boxes=b, labels=cls) for b, cls in zip(preds, preds_cls)],
        [dict(boxes=b, labels=cls) for b, cls in zip(target, target_cls)],
        word_grids, row_grids
    )
    rez = grid_metric.compute()
    return {f'GridIoU_{k}': v for k, v in rez.items()}


def fun_result_to_row(train_result, test_result):
    return test_result


if __name__ == '__main__':
    exp = Experimenter(name=EXP_NAME, result_save_path=exp_path)

    params = BASE_PARAMS.copy()
    dict_params = {
        EXP_NAME: {
            'model_param': params,
            'dataset_param': {},
            'train_param': params,
            'test_param': params,
        }
    }

    exp.experiment(
        fun_get_model_with_param,
        fun_get_dataset_with_param,
        fun_train_model_with_param,
        fun_test_model_with_param,
        fun_result_to_row,
        dict_params
    )
