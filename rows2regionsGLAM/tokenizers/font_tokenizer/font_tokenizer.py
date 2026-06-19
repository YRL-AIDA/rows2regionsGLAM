from ..base_line_tokenizer import RowGLAMTokenizer as BaseLineTokenizer
from pager.page_model.sub_models.dtype import ImageSegment, Font
from typing import Dict, List
import numpy as np

NULL_FONT = []


class RowGLAMTokenizer(BaseLineTokenizer):
    def get_dict_vec(self) -> Dict[str, List[int]]:
        dict_feature = super().get_dict_vec()
        list_feature = []
        for list_f in dict_feature.values():
            list_feature += list_f
        N = len(list_feature)
        dict_feature['font_width'] = [N+0]
        dict_feature['font_italic'] = [N+1], 
        dict_feature['font_size'] = [N+2]
        return dict_feature

    def get_name(self) -> str:
        return "Font Tokenizer"

    def get_node_features(self, rows_json, pdf_img):
        if len(rows_json) == 0:
            return [[]]
        rows_texts = [r['text'] for r in rows_json]
        dot_vec = np.array([[1.0 if dot in r else 0.0 for dot in (".", ",", ";", ":")] for r in rows_texts])
        
        list_ind_vec = np.array([self.get_vec_list(r) for r in rows_texts])
        super_vec = np.array([self.get_vec_supper(r) for r in rows_texts])
        coord_vec = np.array([self.get_vec_coord(r_json) for r_json in rows_json])
        heuristics_vec = np.array([self.get_vec_heuristics(r_json) for r_json in rows_json])
        font_feature_vec = np.array([self.get_vec_font(r_json, pdf_img) for r_json in rows_json])
        nodes_feature = np.concat([coord_vec,  dot_vec, super_vec, list_ind_vec, heuristics_vec, font_feature_vec], axis=1)
        return nodes_feature.tolist()
    
    def get_vec_font(self, row, pdf_img):
        # seg = ImageSegment(dict_2p=row['segment'])
        # TODO: проверку сделать
        font = Font(row['font'])
        
        return [font.width, font.italic, font.size]