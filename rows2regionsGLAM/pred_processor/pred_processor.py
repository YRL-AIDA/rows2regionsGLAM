from typing import Dict, List
from pathlib import Path
from ..utils.pdf_manager import PDFManager
from ..utils.row_manager import RowManager
from ..utils.intersect_util import get_num_regions_of_rows
from ..tokenizers import RowGLAMTokenizer
from pager.page_model.sub_models.dtype import ImageSegment


class PredProcessor:
    def __init__(self, **kwargs):
        if 'loger' in kwargs:
            self.loger = kwargs['loger']
        else:
            raise Exception("need loger")
        if 'pdf_manager' in kwargs:
            self.pdf_manager = kwargs['pdf_manager']
        else:
            self.pdf_manager = PDFManager(loger=self.loger)
        if 'row_manager' in kwargs:
            self.row_manager = kwargs['row_manager']
        else:
            self.row_manager = RowManager(loger=self.loger)
        if 'tokenizer' in kwargs:
            self.tokenizer = kwargs['tokenizer']
        else:
            self.tokenizer = RowGLAMTokenizer()
        
        
        self.coco_manager = None
        if 'coco_manager' in kwargs:
            self.coco_manager = kwargs['coco_manager']

    def _get_true_edges(self, token, rows, region_segs, region_categories):
        def is_one_region(num_reg1, num_reg2):
            if num_reg1 == None:
                return 0
            if num_reg2 == None:
                return 0
            if num_reg1 == num_reg2:
                return 1
            return 0

        def get_category(seg, region_segs, region_categories):
            for r, c in zip(region_segs, region_categories):
                if seg.is_intersection(r):
                    return c
            return None

        def get_mini_seg(r):
            img_seg = ImageSegment(dict_2p=r)
            if img_seg.height < 5:
                return img_seg
            delta = int(img_seg.height / 5)
            img_seg.y_bottom_right = img_seg.y_bottom_right - delta
            img_seg.y_top_left = img_seg.y_top_left + delta
            return img_seg

        row_segments = [get_mini_seg(row['segment']) for row in rows]
        A = token['inds']
        
        nums_regions = get_num_regions_of_rows(region_segs, row_segments)
        true_edges = [is_one_region(nums_regions[i], nums_regions[j]) for i, j in zip(A[0], A[1])]
        true_nodes = [get_category(row_seg, region_segs, region_categories) for row_seg in row_segments]
        return true_edges, true_nodes
   

    def get_json_and_img(self, path_pdf):
        pdf_json, pdf_img = self.pdf_manager.get_json_and_img_from_pdf(path_pdf, num_page=0)
        w, h = pdf_json['width'], pdf_json['height']
        row_json = self.row_manager.get_row_json_from_pdf_json(pdf_json)
        return {
            "rows": row_json,
            "width": w,
            "height": h
        }, pdf_img

    def __call__(self, path_pdf) -> Dict:
        path_pdf = Path(path_pdf)
        json, img = self.get_json_and_img(path_pdf)
        torch_dict = self.tokenizer(json['rows'], img)

        if self.coco_manager:
            true_regions, true_category = self.coco_manager(path_pdf.name, json)   
            true_edges, true_nodes = self._get_true_edges(torch_dict, json['rows'], true_regions, true_category) 
            torch_dict['true_edges'] = true_edges
            torch_dict['true_nodes'] = true_nodes
        return torch_dict