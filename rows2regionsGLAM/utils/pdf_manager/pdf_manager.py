from pager.doc_model import MinerPDFModel, PrecisionPDFModel
from pager import PDF2Img, ImageModel, PDFModel

class PDFManager:
    def __init__(self, **conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        self.loger.time_log()

        if "pdf_reader" not in conf.keys(): 
            conf['pdf_reader'] = "PDFMiner"
        if conf['pdf_reader'] == "PDFMiner":
            self.pdf_reader = MinerPDFModel()
        elif conf['pdf_reader'] == "PrecisionPDF":
            self.pdf_reader = PrecisionPDFModel()
        else:
            raise Exception('Неверный способ чтения ("pdf_reader": "PDFMiner" or "PrecisionPDF")')

        self.loger(f"PDF Reader: {conf['pdf_reader']}")


        self.pdf_model = PDFModel()
        self.img_model = ImageModel()
        self.pdf2img = PDF2Img()

    def get_json_from_pdf(self, pdf_path, num_page=0):
        self.pdf_reader.read_from_file(pdf_path)
        self.pdf_reader.extract()
        self.pdf_model.from_dict(self.pdf_reader.to_dict()['pages'][num_page])
        return self.pdf_model.to_dict()

    def get_json_and_img_from_pdf(self, pdf_path, num_page=0):
        self.pdf_reader.read_from_file(pdf_path)
        self.pdf_reader.extract()
        self.pdf_model.from_dict(self.pdf_reader.to_dict()['pages'][num_page])
        self.pdf_model.path = pdf_path
        self.pdf_model.num_page = num_page
        self.pdf2img.convert(self.pdf_model, self.img_model)
        return self.pdf_model.to_dict(), self.img_model.img
