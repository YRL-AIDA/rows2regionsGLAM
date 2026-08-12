import signal

from pagerlib.file_input import FileInput
from pagerlib.extractors.page_extractor import PDFIMGExtractor, FontEmbExtractor


class _PDFTimeoutError(Exception):
    pass


def _pdf_alarm_handler(signum, frame):
    raise _PDFTimeoutError("PDF reading timeout (>120s)")


class PDFManager:
    def __init__(self, **conf):
        self.loger = conf.get('loger', None)
        if self.loger:
            self.loger.time_log()

        if "pdf_reader" not in conf.keys(): 
            conf['pdf_reader'] = "PDFMiner"
        if conf['pdf_reader'] == "PDFMiner":
            self.pdf_reader = FileInput()
        elif conf['pdf_reader'] == "Paddle":
            from .paddle_ocr_reader import PaddleOCRReader
            self.pdf_reader = PaddleOCRReader()
        elif conf['pdf_reader'] == "PrecisionPDF":
            raise Exception('На данный момент PrecisionPDF не реализован')
        else:
            raise Exception('Неверный способ чтения ("pdf_reader": "PDFMiner" or "PrecisionPDF")')

        if self.loger:
            self.loger(f"PDF Reader: {conf['pdf_reader']}")

        self.img_extract = PDFIMGExtractor()
    

    def get_rows(self, row):
        if row.data and 'font_vec' in row.data:
            row.data['font_vec'] = list(row.data['font_vec'])
        row.segment.x_top_left = row.segment.x_top_left 
        row.segment.y_top_left = row.segment.y_top_left - 5 
        row.segment.x_bottom_right = row.segment.x_bottom_right 
        row.segment.y_bottom_right = row.segment.y_bottom_right + 5 
        return row.to_dict()

    def read_pdf(self, pdf_path):
        old_handler = signal.signal(signal.SIGALRM, _pdf_alarm_handler)
        signal.alarm(120)
        try:
            result = self.pdf_reader(pdf_path)
            return result
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def get_json_from_pdf(self, pdf_path, num_page=0):
        prdf = self.read_pdf(pdf_path)
        page = prdf.data['pages'][num_page]
        return {"rows": [self.get_rows(row) for reg in page.children if not reg.children is None for row in reg.children],
                "width": page.segment.width,
                "height": page.segment.height,
                "images": [reg.to_dict() for reg in page.children[:] if reg.children is None]}

    def get_json_and_img_from_pdf(self, pdf_path, num_page=0):
        try:
            prdf = self.read_pdf(pdf_path)
            self.img_extract.extract(prdf)
            page = prdf.data['pages'][num_page]
        except _PDFTimeoutError:
            with open('failed_pdfs.log', 'a') as log:
                log.write(f"TIMEOUT\t{pdf_path}\n")
            print(f"TIMEOUT >120s: {pdf_path}")
            return {}, None
        except Exception as e:
            with open('failed_pdfs.log', 'a') as log:
                log.write(f"ERROR\t{pdf_path}\t{e}\n")
            print('error in PDFManager', e)
            print(pdf_path)
            return {}, None
        return {
            "rows": [self.get_rows(row) for reg in page.children[1:] if not reg.children is None for row in reg.children],
            "width": page.segment.width,
            "height": page.segment.height,
            "images": [reg.to_dict() for reg in page.children[1:] if reg.children is None]
        }, page.children[0].data['array']
        
        
