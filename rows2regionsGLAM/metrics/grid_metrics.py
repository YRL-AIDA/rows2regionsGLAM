from pager import ImageSegment
import numpy as np
from ..utils.intersect_util import get_num_regions_of_rows
def gridIoU(bbox_1:ImageSegment, bbox_2:ImageSegment, grid_blocks:list[ImageSegment]):
    
    # Строки из соседних блоков могут задевать и попадать в true, но в predict не попадают
    out_1, out_2, in_12 = 0, 0, 0
    for el in grid_blocks:
        in_1 = el.is_intersection(bbox_1)
        in_2 = el.is_intersection(bbox_2)
        if in_1 and in_2:
            in_12 += 1
        elif in_1:
            out_1 += 1
        elif in_2:
            out_2 += 1
    all_12 = out_1+out_2+in_12
    if all_12 == 0:
        raise Exception('block is not from grids')
    return in_12/all_12
        

def grid_Precision_and_Recall(bboxes_pred:list[ImageSegment], bboxes_true:list[ImageSegment], grid_blocks:list[ImageSegment], threshold=0.5):
    intersect_matrix_true = get_num_regions_of_rows(bboxes_true, grid_blocks)
    intersect_matrix_pred = get_num_regions_of_rows(bboxes_pred, grid_blocks)
    def get_iou(it, ip):
        in_ip = intersect_matrix_pred == ip
        in_it = intersect_matrix_true == it

        dem =(in_ip | in_it).sum()
        num = (in_ip & in_it).sum()
        
        return 0.0 if dem == 0 else float(num/dem)
    
    res = [[(get_iou(it, ip), it) for it, _ in enumerate(bboxes_true)]  for ip, _ in enumerate(bboxes_pred)]
    res = [sorted([r for r in p if r[0]>threshold], key=lambda r: r[0], reverse=True) for p in res]

    t_no = []
    TP = 0
    for p in res:
        for r in p:
            it = r[1]
            if it is t_no:
               continue
            t_no.append(it)
            TP+=1
     
    TP_pl_FP  = len(bboxes_pred) 
    TP_pl_FN  = len(bboxes_true)
    precision = TP/TP_pl_FP 
    recall    = TP/TP_pl_FN
    return precision, recall

class MultiGridMetric:
    def __init__(self):
        self.grid_metrics = dict()

    def update(self, preds_dicts, target_dicts, row_grids, word_grids):
        
        labels = set([label for preds_dict in preds_dicts for label in preds_dict['labels']]) | set([label for target_dict in target_dicts for label in target_dict['labels']])
        for label in labels:
            grid_metric = GridMetric()
            grid_metric.update(
                preds=[[p for p, l in zip(preds_dict['boxes'], preds_dict['labels']) if l == label] for preds_dict in preds_dicts],
                target=[[t for t, l in zip(target_dict['boxes'], target_dict['labels']) if l == label] for target_dict in target_dicts],
                row_grids=row_grids, word_grids=word_grids
            )
            self.grid_metrics[label] = grid_metric
        
        grid_metric = GridMetric()
        grid_metric.update(
                preds=[preds_dict['boxes'] for preds_dict in preds_dicts],
                target=[target_dict['boxes'] for target_dict in target_dicts],
                row_grids=row_grids, word_grids=word_grids
        )
        self.grid_metrics['all'] = grid_metric

    def compute(self):
        rezs = dict()
        for label, grid_metric in self.grid_metrics.items():
            rez = grid_metric.compute()
            for th, metrics in rez.items():
                rezs[f'{th} ({label})'] = metrics['f1_row']
        return rezs

class GridMetric:
    def __init__(self, box_format="xywh"):
        self.coef = [0.5, 0.95]
        self.arrays = {
            f"threshold_{coef}": {
                "all_precision_row": [],
                "all_recall_row": [],
                "all_precision_word": [],
                "all_recall_word": [],
                "all_f1_word": [],
                "all_f1_row": []
            } for coef in self.coef
        }

        self.rez = {
            f"threshold_{coef}": {
                "precision_row": None,
                "recall_row": None,
                "f1_row": None,
                "precision_word": None,
                "recall_word": None,
                "f1_word": None
            } for coef in self.coef
        }
        if box_format not in ["xywh", "xyxy"]:
            raise Exception('Не верный формат ("xywh", "xyxy")')
        self.box_format = box_format

    def update(self, preds, target, row_grids, word_grids):
        i = 0
        N = len(preds)

        if self.box_format == "xywh":
            def get_segs(blocks):
                return [ImageSegment(x_top_left=block[0], 
                                     y_top_left=block[1],
                                     x_bottom_right=block[0]+block[2],
                                     y_bottom_right=block[1]+block[3]) for block in blocks]
        elif self.box_format == "xyxy":
            def get_segs(blocks):
                return [ImageSegment(x_top_left=block[0], 
                                     y_top_left=block[1],
                                     x_bottom_right=block[2],
                                     y_bottom_right=block[3]) for block in blocks]


        for bboxes_pred, bboxes_true, grid_row, grid_word in zip(preds, target, row_grids, word_grids):
            i += 1
            try:
                seg_bboxes_pred = get_segs(bboxes_pred)
                seg_bboxes_true = get_segs(bboxes_true)
                seg_grid_row = get_segs(grid_row)
                seg_grid_word = get_segs(grid_word)
                rez = {
                    f"threshold_{coef}":self._get_pair_threshold(seg_bboxes_pred, seg_bboxes_true, seg_grid_row, seg_grid_word, coef)
                    for coef in self.coef
                } 
                for th, th_rez in self.arrays.items():
                    loc_th_rez = rez[th]
                    for metric, ans in loc_th_rez.items():
                        th_rez["all_"+metric].append(ans)
                print(f"{(i) / N * 100:4.2f} %", end='\r')
            except Exception as e:
                print(e)
        self.__update()
    
    def compute(self):
        return self.rez

        
        
    def _get_pair_threshold(self, bboxes_pred, bboxes_true, grid_row, grid_word, threshold):
        if len(bboxes_pred) == 0 and len(bboxes_true) == 0:
            precision_row, recall_row, precision_word, recall_word = 1, 1, 1, 1
        elif len(bboxes_pred) == 0:
            precision_row,  precision_word = 1, 1
            recall_row, recall_word = 0, 0 
        elif len(bboxes_true) == 0:
            precision_row,  precision_word = 0, 0
            recall_row, recall_word = 1, 1
        else:
            precision_row, recall_row = grid_Precision_and_Recall(bboxes_pred, bboxes_true, grid_row, threshold)
            precision_word, recall_word = grid_Precision_and_Recall(bboxes_pred, bboxes_true, grid_word, threshold)
        return {
            "precision_row": precision_row,
            "recall_row": recall_row,
            "f1_row": 2*precision_row*recall_row/(precision_row+recall_row) if precision_row+recall_row > 0 else 0,
            "precision_word": precision_word,
            "recall_word": recall_word,
            "f1_word": 2*precision_word*recall_word/(precision_word+recall_word) if precision_word+recall_word > 0 else 0
        }


    
    def __update(self):
        for th, th_rez in self.arrays.items():
            p_r = np.mean(th_rez["all_precision_row"])
            p_w = np.mean(th_rez["all_precision_word"])
            r_r = np.mean(th_rez["all_recall_row"])
            r_w = np.mean(th_rez["all_recall_word"])
            f_r = np.mean(th_rez["all_f1_row"])
            f_w = np.mean(th_rez["all_f1_word"])
            self.rez[th] = {
                "precision_row": p_r,
                "recall_row": r_r,
                "f1_row": f_r,
                "precision_word": p_w,
                "recall_word": r_w,
                "f1_word": f_w
            }
            
        
    def __str__(self):
        str_ = "="*50+"\n"
        for th, th_rez in self.rez.items():
            str_ += th + "-"*20 + "\n"
            for metric, value in th_rez.items():
                str_ += f"{metric:<20}:{value:.4f}\n"
        return str_
        
