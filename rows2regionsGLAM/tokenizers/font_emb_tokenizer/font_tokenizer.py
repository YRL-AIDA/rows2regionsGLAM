from ..font_tokenizer.base import FontRowGlAMTokenizer
from pagerlib.extractors.page_extractor.font_emb_extractor.font_identifier import load_model, row_to_vec
from pagerlib.dtypes import ImageSegment
import cv2
from PIL import Image
import numpy as np

SIZES = (512, 32, 16)


class RowGLAMTokenizer(FontRowGlAMTokenizer):

    def __init__(self, size=512):
        if size not in SIZES:
            raise ValueError(f"Size must be one of {SIZES}, got {size}")
        self._size = size
        self._model = None
        super().__init__()

    @property
    def _lazy_model(self):
        if self._model is None:
            self._model = load_model(self._size)
        return self._model

    def get_vec_font(self, row, pdf_img):
        seg = ImageSegment(dict_p_size=row['segment'])
        row_img = seg.get_segment_from_img(pdf_img)
        row_cv2 = cv2.cvtColor(row_img, cv2.COLOR_RGB2GRAY)
        pil_image = Image.fromarray(row_cv2)
        return row_to_vec(self._lazy_model, pil_image).numpy()

    def get_num_font_features(self) -> int:
        return self._size
