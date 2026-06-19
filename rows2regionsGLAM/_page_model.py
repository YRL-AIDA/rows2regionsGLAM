from abc import ABC, abstractmethod
from typing import List, Dict
from collections import Counter
import numpy as np
from pagerlib.dtypes import ImageSegment, Region, Row


class BaseConverter(ABC):
    @abstractmethod
    def convert(self, input_model, output_model, pdf_img):
        pass


class _BaseModel:
    def __init__(self):
        self._data = {}

    def from_dict(self, dict_: Dict):
        self._data = dict_.copy()

    def to_dict(self) -> Dict:
        return self._data.copy()


class RowsModel(_BaseModel):
    def __init__(self):
        super().__init__()
        self._data = {"rows": []}


class RegionModel(_BaseModel):
    def __init__(self):
        super().__init__()
        self._data = {"regions": []}


class ImageModel:
    def __init__(self):
        self.img = None

    def show(self):
        import matplotlib.pyplot as plt
        if self.img is not None:
            plt.imshow(self.img)
            plt.show()


def _merge_segment(segs: List[ImageSegment]) -> List[int]:
    def _for(array_segs, array_ind):
        for ki, i in enumerate(array_ind):
            for kj, j in enumerate(array_ind):
                if i != j and array_segs[ki].is_intersection(array_segs[kj]):
                    array_segs[ki].set_segment_max_segments([array_segs[kj], array_segs[ki]])
                    array_segs[kj] = array_segs[ki]
                    array_ind[kj] = array_ind[ki]
                    return True
        return False

    array_ind = [i for i in range(len(segs))]
    array_segs = [ImageSegment(dict_2p=seg.get_segment_2p()) for seg in segs]
    change = True
    while change:
        change = _for(array_segs, array_ind)
    new_ = dict()
    for i in array_ind:
        if i not in new_:
            new_[i] = len(new_.keys())
    array_ind = [new_[i] for i in array_ind]
    return array_ind


class MergeExtractor:
    def extract(self, region_model: RegionModel):
        regions = region_model.to_dict().get("regions", [])
        if not regions:
            return

        segs = [r.segment for r in regions]

        index_segs = _merge_segment(segs)

        new_segs: Dict[int, List[int]] = dict()
        for ind_seg, ind_new_seg in enumerate(index_segs):
            if ind_new_seg in new_segs:
                new_segs[ind_new_seg] = new_segs[ind_new_seg] + [ind_seg]
            else:
                new_segs[ind_new_seg] = [ind_seg]

        merged_regions = []
        for indices in new_segs.values():
            merged_rows = []
            for idx in indices:
                merged_rows.extend(regions[idx].children)
            labels = [regions[i].data.get('label') for i in indices if regions[i].data and 'label' in regions[i].data]
            label = Counter(sorted(labels)).most_common(1)[0][0] if labels else None
            merged_regions.append(Region(children=merged_rows, data={'label': label}))

        region_model.from_dict({"regions": merged_regions})
