from ..base_line_tokenizer import RowGLAMTokenizer as BaseLineTokenizer
from typing import Dict, List
from abc import abstractmethod
import numpy as np


class FontRowGlAMTokenizer(BaseLineTokenizer):

    def get_dict_vec(self) -> Dict[str, List[int]]:
        dict_feature = super().get_dict_vec()
        list_feature = []
        for list_f in dict_feature.values():
            list_feature += list_f
        N = len(list_feature)
        dict_feature['font_width'] = [N + 0]
        dict_feature['font_italic'] = [N + 1]
        dict_feature['font_size'] = [N + 2]
        return dict_feature

    def get_name(self) -> str:
        return "Font Tokenizer"

    def get_node_features(self, rows_json, pdf_img):
        old_feature = np.array(super().get_node_features(rows_json, pdf_img))
        font_feature_vec = np.array([self._get_vec_font_safe(r_json, pdf_img) for r_json in rows_json])
        nodes_feature = np.concat([old_feature, font_feature_vec], axis=1)
        return nodes_feature.tolist()

    def _get_vec_font_safe(self, row, pdf_img):
        try:
            vec = self.get_vec_font(row, pdf_img)
            return vec
        except (KeyError, Exception):
            return np.zeros(self.get_num_font_features(), dtype=np.float32)

    @abstractmethod
    def get_vec_font(self, row, pdf_img):
        pass

    @abstractmethod
    def get_num_font_features(self) -> int:
        pass
