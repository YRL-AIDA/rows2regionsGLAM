from ..font_tokenizer.base import FontRowGlAMTokenizer
from pagerlib.dtypes import ImageSegment
from pagerlib.extractors.page_extractor.font_emb_extractor.font_identifier import load_model, row_to_vec
import cv2
from PIL import Image
import numpy as np

FONT_EMB_DIM = 512


class RowGLAMTokenizer(FontRowGlAMTokenizer):

    def __init__(self):
        self.model = load_model()
        super().__init__()

    def get_vec_font(self, row, pdf_img):
        seg = ImageSegment(dict_p_size=row['segment'])
        row_img = seg.get_segment_from_img(pdf_img)
        row_cv2 = cv2.cvtColor(row_img, cv2.COLOR_RGB2GRAY)
        pil_image = Image.fromarray(row_cv2)
        vec = row_to_vec(self.model, pil_image).numpy()
        return vec

    def get_num_font_features(self) -> int:
        return FONT_EMB_DIM
