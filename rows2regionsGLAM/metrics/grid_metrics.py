from pager import ImageSegment
import numpy as np

def gridIoU(bbox_1:ImageSegment, bbox_2:ImageSegment, grid_blocks:list[ImageSegment]):
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
    def get_iou(bp, bt, grid_blocks):
        try:
            return gridIoU(bp, bt, grid_blocks)
        except:
            return 0.0
    mtrx = np.array([
        [get_iou(bp, bt, grid_blocks) for bt in bboxes_true] for bp in bboxes_pred 
    ])
    TP = int(sum(mtrx.max(axis=0)>threshold))
    TP_pl_FP = len(bboxes_pred) 
    TP_pl_TN = len(bboxes_true)
    precision = TP/TP_pl_FP 
    recall = TP/TP_pl_TN
    return precision, recall


class GridMetric:
    def __init__(self, box_format="xywh"):
        self.arrays = {
            "threshold_05": {
                "all_precision_row": [],
                "all_recall_row": [],
                "all_precision_word": [],
                "all_recall_word": []
            }, 
            "threshold_95": {
                "all_precision_row": [],
                "all_recall_row": [],
                "all_precision_word": [],
                "all_recall_word": []
            }, 
        }

        self.rez = {
            "threshold_05": {
                "precision_row": None,
                "recall_row": None,
                "f1_row": None,
                "precision_word": None,
                "recall_word": None,
                "f1_word": None
            }, 
            "threshold_95": {
                "precision_row": None,
                "recall_row": None,
                "f1_row": None,
                "precision_word": None,
                "recall_word": None,
                "f1_word": None
            },  
        }
        if box_format not in ["xywh", "xyxy"]:
            raise Exception('Не верный формат ("xywh", "xyxy")')
        self.box_format = box_format

    def add_pair(self, bboxes_pred, bboxes_true, grid_row, grid_word):
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
        seg_bboxes_pred = get_segs(bboxes_pred)
        seg_bboxes_true = get_segs(bboxes_true)
        seg_grid_row = get_segs(grid_row)
        seg_grid_word = get_segs(grid_word)
        rez = {
            "threshold_05":self._get_pair_threshold(seg_bboxes_pred, seg_bboxes_true, seg_grid_row, seg_grid_word, 0.50),
            "threshold_95":self._get_pair_threshold(seg_bboxes_pred, seg_bboxes_true, seg_grid_row, seg_grid_word, 0.95)
        } 
        for th, th_rez in self.arrays.items():
            loc_th_rez = rez[th]
            for metric, ans in loc_th_rez.items():
                th_rez["all_"+metric].append(ans)
                
        
        
    def _get_pair_threshold(self, bboxes_pred, bboxes_true, grid_row, grid_word, threshold):
        precision_row, recall_row = grid_Precision_and_Recall(bboxes_pred, bboxes_true, grid_row, threshold)
        precision_word, recall_word = grid_Precision_and_Recall(bboxes_pred, bboxes_true, grid_word, threshold)

        return {
            "precision_row": precision_row,
            "recall_row": recall_row,
            "precision_word": precision_word,
            "recall_word": recall_word
        }


    
    def update(self):
        for th, th_rez in self.arrays.items():
            p_r = np.mean(th_rez["all_precision_row"])
            p_w = np.mean(th_rez["all_precision_word"])
            r_r = np.mean(th_rez["all_recall_row"])
            r_w = np.mean(th_rez["all_recall_word"])
            self.rez[th] = {
                "precision_row": p_r,
                "recall_row": r_r,
                "f1_row": 2*p_r*r_r/(p_r+r_r),
                "precision_word": p_w,
                "recall_word": r_w,
                "f1_word": 2*p_w*r_w/(p_w+r_w)
            }
            
        
    def __str__(self):
        str_ = "="*50+"\n"
        for th, th_rez in self.rez.items():
            str_ += th + "-"*20 + "\n"
            for metric, value in th_rez.items():
                str_ += f"{metric:<20}:{value:.4f}\n"
        return str_
        
