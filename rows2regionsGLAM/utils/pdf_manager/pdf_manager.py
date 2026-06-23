# from pager.doc_model import MinerPDFModel, PrecisionPDFModel
# from pager import PDF2Img, ImageModel, PDFModel

from pagerlib.file_input import FileInput
from pagerlib.extractors.page_extractor import PDFIMGExtractor, FontEmbExtractor
from multiprocessing import Process, Queue


class PDFManager:
    def __init__(self, **conf):
        self.loger = conf.get('loger', None)
        if self.loger:
            self.loger.time_log()

        if "pdf_reader" not in conf.keys(): 
            conf['pdf_reader'] = "PDFMiner"
        if conf['pdf_reader'] == "PDFMiner":
            self.pdf_reader = FileInput()
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
        
        return row.to_dict()

    def read_pdf(self, pdf_path):        
        # result_queue = Queue()

        # def pdf_reader_wrapper(pdf_path, queue):
        #     result = self.pdf_reader(pdf_path) 
        #     queue.put(result) 
        # p = Process(target=pdf_reader_wrapper, args=(pdf_path, result_queue))
        # p.start()
        # p.join(timeout=30)  # Ожидаем выполнение не более 10 секунд
        
        # result = None
        # if p.is_alive():
        #     # Если процесс все еще работает - принудительно завершаем
        #     p.terminate()
        #     p.join()  # Ждем завершения terminate
        #     print(f"Функция read_pdf для файла {pdf_path} превысила лимит времени 30 секунд")
        # else:
        #     # Если процесс успешно завершился, получаем результат
        #     try:
        #         result = result_queue.get_nowait()
        #     except:
        #         # Очередь пуста - что-то пошло не так
        #         pass
        result = self.pdf_reader(pdf_path) 
        return result
    
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
        except Exception as e:
            print('error in PDFManager', e)
            print(pdf_path)
            return {}, None
        return {
            "rows": [self.get_rows(row) for reg in page.children[1:] if not reg.children is None for row in reg.children],
            "width": page.segment.width,
            "height": page.segment.height,
            "images": [reg.to_dict() for reg in page.children[1:] if reg.children is None]
        }, page.children[0].data['array']
        
        
