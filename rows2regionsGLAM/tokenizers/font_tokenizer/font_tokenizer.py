from .base import FontRowGlAMTokenizer
from pagerlib.dtypes.physical_elements.font import Font


class RowGLAMTokenizer(FontRowGlAMTokenizer):

    def get_vec_font(self, row, pdf_img):
        font = Font(row['font'])
        return [font.width, font.italic, font.size]

    def get_num_font_features(self) -> int:
        return 3
