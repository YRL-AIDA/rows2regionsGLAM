from pager.page_model.sub_models import BaseConverter, RegionModel, RowsModel
from pager.page_model.sub_models.dtype import ImageSegment, Region, Graph
from pager import MergeExtractor
import numpy as np

class Rows2Regions(BaseConverter):
    def __init__(self, conf):
        self.rows2regionsGLAM_tokenizer = conf['tokenizer']# manager_model.get_model("rowGLAM-tokenizer")
        self.rows2regionsGLAM = conf['model']#
        self.is_merge_extract =  conf['is_merge_extract']
        self.merge_extract = MergeExtractor()
        self.classes = conf['classes']

        
    def convert(self, input_model: RowsModel, output_model: RegionModel, pdf_img):
        page_json = input_model.to_dict()
        region_list = self.get_region(page_json['rows'], pdf_img)
        output_model.from_dict({"regions": region_list})

        if self.is_merge_extract:
            self.merge_extract.extract(output_model)
        # сортировка после создания региона
        # sorter = RegionSorterCutXYExtractor()
        # sorter.extract(output_model)

    def get_region(self, rows_json, pdf_img):
        graph_dict_torch = self.rows2regionsGLAM_tokenizer(rows_json, pdf_img)
        result = self.rows2regionsGLAM(graph_dict_torch)
        result['deleted_edges'] = result['E_pred'] < 0.5
        
        graph = graph_dict_torch['inds']
        deleted_edges = result['deleted_edges']
        node_classes = result['node_classes']
        regions = self.regions_from_graph(rows_json, graph, deleted_edges, node_classes)
        return regions
    

    def regions_from_graph(self, rows_json, graph, deleted_edges, node_classes):
        graph_ = Graph()
        regions = []
        
        for row_json in rows_json:
            segment = ImageSegment(dict_2p=row_json['segment'])
            xc, yc = segment.get_center()
            graph_.add_node(xc, yc)

        for node_i, node_j, ind in zip(graph[0], graph[1], deleted_edges):
            if not ind:
                graph_.add_edge(node_i+1, node_j+1)

        for reg in graph_.get_related_graphs():
            indexes = [node.index-1 for node in reg.get_nodes()]
            row_classes  = np.array([node_classes[i].detach().numpy() for i in indexes])
            lable = self.classes[np.argmax(row_classes.mean(axis=0))]
            regions.append({'rows': [rows_json[i] for i in indexes], 'label': lable})
        return regions