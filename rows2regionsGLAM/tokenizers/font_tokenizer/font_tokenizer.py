from .base import FontRowGlAMTokenizer
from pagerlib.dtypes.physical_elements.font import Font
import numpy as np


class RowGLAMTokenizer(FontRowGlAMTokenizer):

    def get_vec_font(self, row, pdf_img):
        """Extract aggregated font features from all words in a row.

        Walks over row['words'][i]['font'], creates a Font object per word,
        and aggregates: width → max, italic → max, size → np.median.
        """
        words = row.get('words', [])
        if not words:
            return [0.0, 0.0, -1.0]

        widths = []
        italics = []
        sizes = []

        for w in words:
            font_dict = w.get('font') if isinstance(w, dict) else None
            if not isinstance(font_dict, dict):
                continue
            try:
                f = Font(font_dict)
            except Exception:
                continue
            widths.append(f.width)
            italics.append(f.italic)
            sizes.append(f.size)

        if not widths:
            return [0.0, 0.0, -1.0]

        width = float(max(widths))
        italic = float(max(italics))

        valid_sizes = [s for s in sizes if s != -1]
        if valid_sizes:
            size = float(np.median(valid_sizes))
        else:
            size = -1.0

        return [width, italic, size]

    def get_num_font_features(self) -> int:
        return 3
