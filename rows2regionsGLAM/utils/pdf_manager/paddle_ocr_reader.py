from pathlib import Path
from pagerlib.dtypes import PageRDF, Row, Page, Region, ImageSegment, Word
from pagerlib.file_input import FileInput
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    return_word_box=True,
    engine="paddle",
    
)

class PaddleOCRReader:
    """PDF reader using PaddleOCR for word-level text extraction.

    Callable interface compatible with FileInput: returns a PageRDF
    with pages containing Region -> Row -> Word hierarchy.
    """

    def __init__(self) -> None:
        """Initialize PaddleOCR with optimized settings for word extraction."""
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            return_word_box=True,
            engine="paddle",
        )
        self.pdf_reader = FileInput()
    def __call__(self, pdf_path: str | Path) -> PageRDF:
        return self.read_pdf(pdf_path)
     
    def read_pdf(self, pdf_path: str | Path) -> PageRDF:
            """Read a PDF file using PaddleOCR and return a PageRDF."""
            
            output = self.ocr.predict(input=str(pdf_path))
            pdf_r = self.pdf_reader(str(pdf_path))
            pages = []
            for res, pdf_res in zip(output, pdf_r.data['pages']):
                w, h = pdf_res.segment.width, pdf_res.segment.height
                H, W = res['doc_preprocessor_res']['output_img'].shape[:2]
                norm_w = w/W
                norm_h = h/H
                rows = []
    
                for word_texts, word_boxes in zip(
                    res["text_word"], res["text_word_boxes"]
                ):
                    words = []
                    for w_text, w_box in zip(word_texts, word_boxes):
                        if not w_text.strip():
                            continue
                        x0, y0, x1, y1 = w_box
                        try:
                            seg = ImageSegment(
                                x_top_left=int(norm_w*x0),
                                y_top_left=int(norm_h*y0),
                                x_bottom_right=int(norm_w*x1),
                                y_bottom_right=int(norm_h*y1),
                            )
                        except:
                            continue
                        words.append(Word(segment=seg, data={"text": w_text, "font": None}))
    
                    if words:
                        rows.append(Row(children=words))
    
                region = Region(children=rows)
                page = Page(
                    children=[region],
                    segment=ImageSegment(
                        x_top_left=0,
                        y_top_left=0,
                        x_bottom_right=w,
                        y_bottom_right=h,
                    ),
                )
                pages.append(page)
    
            prdf = PageRDF()
            prdf.data["pages"] = pages
            prdf.data["path"] = str(pdf_path)
            return prdf
