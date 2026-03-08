from pager.page_model.sub_models.dtype import ImageSegment
from typing import List
import numpy as np

def how_much_intersection(reg:ImageSegment, row:ImageSegment):
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


def get_num_regions_of_rows(region_segs: List[ImageSegment], row_segments: List[ImageSegment]):
    nums_regions = []
    for row in row_segments:
        reg_ = [how_much_intersection(reg, row) for reg in region_segs]
        if len(reg_) == 0 or max(reg_) == 0:
            nums_regions.append(None)
        else:
            nums_regions.append(np.argmax(reg_))
    return np.array(nums_regions)