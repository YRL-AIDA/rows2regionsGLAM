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
        pdf_json, img = self.get_json_and_img(path_pdf)
        torch_dict = self.tokenizer(pdf_json['rows'], img)
        
        return {
            "pdf_json": pdf_json,
            "img": img,
            "torch_dict": torch_dict
        }


    