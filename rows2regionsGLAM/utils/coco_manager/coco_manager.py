import json

class COCOManager:
    def __init__(self, conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        self.loger("COCOManager")
        self.loger.time_log()

        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        self.loger("COCOManager")
        if "coco_path" not in conf.keys():
            raise Exception('Создайте и передайте логер "coco_path": path.json)')
        else:
            self.coco_path = conf['coco_path']




    def get_regions_from_json(self):
        coco_path = self.coco_path
        
        with open(coco_path, 'r') as f:
            coco = json.load(f)
        
        pdf_ann = dict() 
        
        id_2_file = {}
        for im in coco['images']:
            name = im['file_name'][:-3]+'pdf'
            img_id = im['id']
            id_2_file[img_id] = name 
            
        
        get_info = lambda an: {'segment': {
            'x_top_left': int(an['bbox'][0]),
            'y_top_left': int(an['bbox'][1]),
            'width':      int(an['bbox'][2]),
            'height':     int(an['bbox'][3])
        }, 
                            
            'category_id': an['category_id']
        }
        
        for an in coco['annotations']:
            pdf_name = id_2_file[an['image_id']]
            if pdf_name in pdf_ann:
                pdf_ann[pdf_name]['regions'].append(get_info(an))
            else:
                pdf_ann[pdf_name] = {'regions': [get_info(an)]}

        self.loger(str(coco['categories']))

        coco_classes = {}
        for cat in coco['categories']:
            coco_classes[cat["id"]] = cat["name"]
        coco_classes[0] = 'other'

        return pdf_ann, coco_classes