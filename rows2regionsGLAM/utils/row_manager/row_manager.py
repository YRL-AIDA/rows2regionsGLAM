from pager import Row, ImageSegment

import numpy as np
class RowManager:
    def __init__(self, **conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        self.loger.time_log()
        self.loger("Create RowManager")

        if "add_image" not in conf.keys(): 
            conf['add_image'] = True
            
        self.is_add_image = conf['add_image']
        self.loger(f"Add image as row: {conf['add_image']}")


    def get_row_array_from_json(self, json_pdf):
        json_rows = self.get_row_json_from_pdf_json( json_pdf)
        return [Row(json_row) for json_row in json_rows]
    
    def get_row_json_from_pdf_json(self, json_pdf):
        if self.is_add_image:
            image_rows = json_pdf['images']
            rows = self.fix_row_this_image(json_pdf['rows'], image_rows)
        return rows
    

    def fix_row_this_image(self, rows, image_rows):
        def get_rows(dict_row):
            dict_row['words'] = [{"segment": w['segment'],
                                 "text": w['data']['text'],
                                 "font": w['data']['font']} for w in dict_row['words']] if 'words' in dict_row else []
            return Row(dict_row=dict_row)
            # r.from_dict(dict_row)
            # return r
        
        img_rows = [get_rows(row) for row in image_rows]
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
        array_rows = [get_rows(row) for row in rows]
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