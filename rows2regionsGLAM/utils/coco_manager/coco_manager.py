import json
import os


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
        if "pdf_manager" not in conf.keys():
            raise Exception('Создайте и передайте pdf_manager')
        else:
            self.pdf_manager = conf['pdf_manager']




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

    def add_region_to_regions(self, item, regions, class_id_map, page_w, page_h, img_w, img_h):
        poly = item["poly"]

        xs = poly[::2]
        ys = poly[1::2]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        # НОРМАЛИЗАЦИЯ
        coef_w, coef_h = img_w / page_w, img_h / page_h
        x_top_left = x_min * coef_w
        y_top_left = y_min * coef_h
        width = (x_max - x_min) * coef_w
        height = (y_max - y_min) * coef_h

        category_type = item["category_type"]
        mapped_category = OMNI2DOC_MAP.get(category_type, 'other')
        category_id = class_id_map.get(mapped_category, 0)

        regions.append({
            "segment": {
                "x_top_left": x_top_left,
                "y_top_left": y_top_left,
                "width": width,
                "height": height
            },
            "category_id": category_id
        })

    def convert_omnidocbench_to_custom_format(self):
        coco_path = self.coco_path
        base_dir = os.path.dirname(coco_path)
        ori_pdfs = os.path.join(base_dir, "ori_pdfs")
        with open(coco_path, 'r', encoding='utf-8') as f:
            omnidocbench_data = json.load(f)

        coco_classes = DOCLAYNET_CLASSES
        result = {}
        class_id_map = {v: k for k, v in DOCLAYNET_CLASSES.items()}

        for page in omnidocbench_data:
            page_info = page["page_info"]
            page_w = page_info["width"]
            page_h = page_info["height"]
            page_filename = page_info.get("image_path", "").split("/")[-1][:-3]+'pdf'
            pdf_json, pdf_img = self.pdf_manager.get_json_and_img_from_pdf(os.path.join(ori_pdfs, page_filename))
            img_w, img_h = pdf_json['width'], pdf_json['height']

            regions = []

            for layout_det in page["layout_dets"]:
                if 'merge_list' in layout_det:
                    for merge_item in layout_det["merge_list"]:
                        self.add_region_to_regions(merge_item, regions, class_id_map, page_w, page_h, img_w, img_h)
                else:
                    self.add_region_to_regions(layout_det, regions, class_id_map, page_w, page_h, img_w, img_h)

            result[page_filename] = {
                "regions": regions
            }

        return result, coco_classes


OMNI2DOC_MAP = {
    'title': 'Title',
    'text_block': 'Text',
    'figure': 'Picture',
    'figure_caption': 'Caption',
    'figure_footnote': 'Footnote',
    'table': 'Table',
    'table_caption': 'Caption',
    'table_footnote': 'Footnote',
    'equation_isolated': 'Formula',
    'equation_caption': 'Caption',
    'header': 'Section-header',
    'footer': 'Page-footer',
    'page_number': 'other',
    'page_footnote': 'Footnote',
    'abandon': 'other',
    'code_txt': 'Text',
    'code_txt_caption': 'Caption',
    'reference': 'Text',
}

# Список классов для Doclaynet
DOCLAYNET_CLASSES = {
    1: 'Caption',
    2: 'Footnote',
    3: 'Formula',
    4: 'Text', # в omni нет list item
    5: 'Page-footer',
    6: 'Page-header',
    7: 'Picture',
    8: 'Section-header',
    9: 'Table',
    10: 'Text',
    11: 'Title',
    0: 'other'
}


