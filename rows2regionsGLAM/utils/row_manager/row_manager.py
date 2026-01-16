from pager import PDFModel, RowsModel, PDF2Rows
from pager.page_model.sub_models.dtype import Row, ImageSegment
import numpy as np
class RowManager:
    def __init__(self, conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        self.loger.time_log()
        self.loger("Create RowManager")

        if "add_image" not in conf.keys(): 
            raise Exception('Добавлять к строкам изображения? ("add_image: True or False")')
        else:
            self.is_add_image = conf['add_image']
            self.loger(f"Add image as row: {conf['add_image']}")

        self.pdf_model = PDFModel()
        self.rows_model = RowsModel()
        self.pdf2row = PDF2Rows()

    def get_row_array_from_json(self, json_pdf):
        json_rows = self.get_row_json_from_pdf_json( json_pdf)
        return [Row(json_row) for json_row in json_rows]
    
    def get_row_json_from_pdf_json(self, json_pdf):
        self.pdf_model.from_dict(json_pdf)
        self.pdf2row.convert(self.pdf_model, self.rows_model)
        rows = self.rows_model.to_dict()['rows']
        if self.is_add_image:
            image_rows = json_pdf['images']
            rows = self.fix_row_this_image(rows, image_rows)
        return rows
    

    def fix_row_this_image(self, rows, image_rows):
        img_rows = [Row(row) for row in image_rows]
        bool_matrix = np.array([
                        [
                            img_row_i.segment.is_intersection(img_row_j.segment)
                        for img_row_j in img_rows] 
                    for img_row_i in img_rows])
        new_bool_matrix = bool_matrix == None
        while (new_bool_matrix != bool_matrix).any():
            new_bool_matrix = bool_matrix.copy()
            bool_matrix = bool_matrix@bool_matrix
                        
        inds = np.array([i for i in range(len(img_rows))])

        blocks = []
        for _ in range(len(inds)):
            if len(inds) < 1:
                break
            i = np.argmin(inds)
            neig = inds[bool_matrix[i]]
            inds = inds[~bool_matrix[i]]
            blocks.append(neig)
            bool_matrix = bool_matrix[~bool_matrix[i], : ][:, ~bool_matrix[i]] 
            
        def get_row(rows):
            img_seg = ImageSegment(0,0,1,1)
            img_seg.set_segment_max_segments([r.segment for r in rows])
            row = Row({'segment': img_seg.get_segment_2p(), 'text':' '})
            return row

        img_rows = [
            get_row([img_rows[int(b)] for b in block])
            for block in blocks
        ]
        new_rows= []
        array_rows = [Row(row) for row in rows]
        for row in array_rows:
            include=True
            for img_row in img_rows:
                if row.segment.is_intersection(img_row.segment):
                    img_row.segment.set_segment_max_segments([img_row.segment, row.segment])
                    include=False
            if include:  
                new_rows.append(row)
        
        rows = new_rows+img_rows       
        return [row.to_dict() for row in rows]