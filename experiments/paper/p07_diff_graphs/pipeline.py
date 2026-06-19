import numpy as np
import torch
from pathlib import Path

from rows2regionsGLAM.tokenizers import RowGLAMTokenizer


class TrueModel:
    def __call__(self, data_graph_dict):
        return {
            "node_classes": data_graph_dict["true_nodes"],
            "E_pred": data_graph_dict["true_edges"],
        }


class AllRowGLAMTokenizer(RowGLAMTokenizer):
    def get_A(self, rows_json):
        N = len(rows_json)
        A1, A2 = [], []
        for a1 in range(N):
            if a1 == N:
                continue
            for a2 in range(a1 + 1, N):
                A1.append(a1)
                A2.append(a2)
        index = np.argsort(A1)
        A1_ = [A1[i] for i in index]
        A2_ = [A2[i] for i in index]
        return [A1_, A2_]


def get_tokenizer(name_tok):
    if name_tok == "glam":
        return RowGLAMTokenizer()
    elif name_tok == "all":
        return AllRowGLAMTokenizer()


class DiffGraphsPipeline:
    def __init__(self, pred, model, dataset_name2id, model_id2name, coco_manager, default_index, num_classes):
        from rows2regionsGLAM.pipeline.converters import Rows2Regions

        self.pred = pred
        self.model = model
        self.coco_manager = coco_manager
        self.default_index = default_index
        self.num_classes = num_classes
        self.name2id = dataset_name2id
        self.rows2regions = Rows2Regions({
            "model": model,
            "tokenizer": pred.tokenizer,
            "is_merge_extract": True,
            "classes": model_id2name,
        })

    def __call__(self, path):
        from pagerlib.dtypes import ImageSegment
        from rows2regionsGLAM.utils.intersect_util import get_num_regions_of_rows

        path = Path(path)
        rez = self.pred(path)
        torch_dict = rez["torch_dict"]
        pdf_json = rez["pdf_json"]
        rows_json = pdf_json["rows"]

        true_regions, true_categories = self._get_annotations(path.name, pdf_json)
        true_edges, true_nodes = self._compute_true_edges_and_nodes(
            torch_dict, rows_json, true_regions, true_categories
        )

        data = dict(torch_dict)
        data["true_edges"] = torch.tensor([0 if i is None else i for i in true_edges], dtype=torch.float32)
        data["true_nodes"] = self._onehot(true_nodes)

        result = self.rows2regions.rows2regionsGLAM(data)
        result["deleted_edges"] = result["E_pred"] < 0.5

        graph = data["inds"]
        deleted_edges = result["deleted_edges"].cpu()
        node_classes = result["node_classes"].cpu()
        regions = self.rows2regions.regions_from_graph(rows_json, graph, deleted_edges, node_classes)

        regs = []
        for r in regions:
            r.data['label'] = self.name2id[r.data['label']]
            d = r.to_dict()
            d['label'] = d['data']['label']
            regs.append(d)
        return {"regions": regs}

    def _get_annotations(self, name_pdf, page_info):
        try:
            return self.coco_manager(name_pdf, page_info)
        except KeyError:
            return [], []

    def _onehot(self, classes):
        vecs = []
        for c in classes:
            v = [0.0] * self.num_classes
            if c is None:
                v[self.default_index] = 1.0
            else:
                v[c] = 1.0
            vecs.append(v)
        return torch.tensor(vecs, dtype=torch.float32)

    def _compute_true_edges_and_nodes(self, token, rows, region_segs, region_categories):
        from pagerlib.dtypes import ImageSegment
        from rows2regionsGLAM.utils.intersect_util import get_num_regions_of_rows

        def get_mini_seg(r):
            img_seg = ImageSegment(dict_p_size=r)
            if img_seg.height < 5:
                return img_seg
            delta = int(img_seg.height / 5)
            img_seg.y_bottom_right = img_seg.y_bottom_right - delta
            img_seg.y_top_left = img_seg.y_top_left + delta
            return img_seg

        def get_category(seg, region_segs, region_categories):
            for r, c in zip(region_segs, region_categories):
                if seg.is_intersection(r):
                    return c
            return None

        def is_one_region(num_reg1, num_reg2):
            if num_reg1 is None or num_reg2 is None:
                return 0
            if num_reg1 == num_reg2:
                return 1
            return 0

        row_segments = [get_mini_seg(row["segment"]) for row in rows]
        A = token["inds"]
        nums_regions = get_num_regions_of_rows(region_segs, row_segments)
        true_edges = [is_one_region(nums_regions[i], nums_regions[j]) for i, j in zip(A[0], A[1])]
        true_nodes = [get_category(row_seg, region_segs, region_categories) for row_seg in row_segments]
        return true_edges, true_nodes
