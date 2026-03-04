from pager.page_model.sub_models.dtype import ImageSegment
import numpy as np

class Cacher:
    def __init__(self, conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        if "pdf_manager" not in conf.keys():
            raise Exception('Создайте и передайте pdf_manager')
        else:
            self.pdf_manager = conf['pdf_manager']
        if "row_manager" not in conf.keys():
            raise Exception('Создайте и передайте row_manager')
        else:
            self.row_manager = conf['row_manager']
        if "tokenizer" not in conf.keys():
            raise Exception('Создайте и передайте tokenizer')
        else:
            self.tokenizer = conf['tokenizer']
        self.loger.time_log()
        self.loger("Create Cacher")

    def _get_true_edges(self, token, rows, region_segs, region_categories):
            
        def how_much_intersection(reg, row):
            if not reg.is_intersection(row):
                return 0.0
            xs = [row.x_top_left, row.x_bottom_right, reg.x_top_left, reg.x_bottom_right]
            W = list(set(xs))
            W.sort()

            ys = [row.y_top_left, row.y_bottom_right, reg.y_top_left, reg.y_bottom_right]
            H = list(set(ys))
            H.sort()

            def get_border(set_list, origin_list):
                if len(set_list) == 4:
                    return set_list[1], set_list[2]
                if len(set_list) == 2:
                    return set_list[0], set_list[1]
                if len(set_list) == 3:
                    if origin_list[0] == origin_list[2]:
                        return set_list[0], set_list[1]
                    else:
                        return set_list[1], set_list[2]
                
            ix0, ix1 = get_border(W, xs)
            iy0, iy1 = get_border(H, ys)
            size_in = (ix1-ix0)*(iy1-iy0)
            size_row = row.height*row.width
            return size_in/size_row





        def is_one_region(num_reg1, num_reg2):
            if num_reg1 == None:
                return 0
            if num_reg2 == None:
                return 0
            if num_reg1 == num_reg2:
                return 1
            return 0

        def get_category(seg, region_segs, region_categories):
            for r, c in zip(region_segs, region_categories):
                if seg.is_intersection(r):
                    return c
            return None

        def get_mini_seg(r):
            img_seg = ImageSegment(dict_2p=r)
            if img_seg.height < 5:
                return img_seg
            delta = int(img_seg.height / 5)
            img_seg.y_bottom_right = img_seg.y_bottom_right - delta
            img_seg.y_top_left = img_seg.y_top_left + delta
            return img_seg

        row_segments = [get_mini_seg(row['segment']) for row in rows]
        A = token['inds']
        nums_regions = []
        for row in row_segments:
            reg_ = [how_much_intersection(reg, row) for reg in region_segs]
            if max(reg_) == 0:
                nums_regions.append(None)
            else:
                nums_regions.append(np.argmax(reg_))
        true_edges = [is_one_region(nums_regions[i], nums_regions[j]) for i, j in zip(A[0], A[1])]
        true_nodes = [get_category(row_seg, region_segs, region_categories) for row_seg in row_segments]
        return true_edges, true_nodes


    def pdf2torch_dict(self, path_pdf, coco_dict_file, name_dataset):
        try:
            pdf_json, pdf_img = self.pdf_manager.get_json_and_img_from_pdf(path_pdf, num_page=0)
            w, h = pdf_json['width'], pdf_json['height']
            row_json = self.row_manager.get_row_json_from_pdf_json(pdf_json)
        except:
            print(path_pdf, '\n')
            return {}
        torch_dict = self.tokenizer(row_json, pdf_img)
        if name_dataset == "doclaynet":
            coef_w, coef_h = w / 1024, h / 1024
        elif name_dataset == "publaynet":
            coef_w, coef_h = 1, 1
        coco_dict_file['regions'] = [r for r in coco_dict_file['regions'] if r['segment']['height'] > 0]
        reg_segments = [ImageSegment(dict_p_size={
            "x_top_left": int(r['segment']['x_top_left'] * coef_w),
            "y_top_left": int(r['segment']['y_top_left'] * coef_h),
            "width": int(r['segment']['width'] * coef_w),
            "height": int(r['segment']['height'] * coef_h)}) for r in coco_dict_file['regions']]
        reg_categories = [r['category_id'] for r in coco_dict_file['regions']]
        true_edges, true_nodes = self._get_true_edges(torch_dict, row_json, reg_segments, reg_categories)

        del torch_dict['sp_A']
        torch_dict['true_edges'] = true_edges
        torch_dict['true_nodes'] = true_nodes

        return torch_dict

    def __call__(self, path_pdf, coco_dict_file, name_dataset):
        return self.pdf2torch_dict(path_pdf, coco_dict_file, name_dataset)