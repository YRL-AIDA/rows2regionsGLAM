import json
from pager.page_model.sub_models.dtype import ImageSegment

class COCOManager:
    def __init__(self, **conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        if "name_dataset" not in conf.keys() or not conf['name_dataset'] in ("doclaynet", "publaynet"):
            raise Exception('Передайте имя датасета "name_dataset": str ("doclaynet", "publaynet")')
        self.name_dataset = conf['name_dataset']

        self.loger("COCOManager")
        self.loger.time_log()

        if "coco_path" not in conf.keys():
            raise Exception('Создайте и путь coco_path": path.json)')
        else:
            self.coco_path = conf['coco_path']
        self.regions, self.classes = self.get_regions_from_json()


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
        self.coco_classes = coco_classes
        return pdf_ann, coco_classes
    
    def __call__(self, name_pdf, page_info):
        if self.name_dataset == "doclaynet":
            coef_w, coef_h = page_info['width'] / 1024, page_info['height'] / 1024
        elif self.name_dataset == "publaynet":
            coef_w, coef_h = 1, 1
        
        regions = self.regions[name_pdf]['regions']
        
        regions = [r for r in regions if r['segment']['height'] > 0]
        reg_segments = [ImageSegment(dict_p_size={
            "x_top_left": int(r['segment']['x_top_left'] * coef_w),
            "y_top_left": int(r['segment']['y_top_left'] * coef_h),
            "width": int(r['segment']['width'] * coef_w),
            "height": int(r['segment']['height'] * coef_h)}) for r in regions]
        reg_categories = [r['category_id'] for r in regions]
        return reg_segments, reg_categories
        