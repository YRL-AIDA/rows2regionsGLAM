from pager.page_model.sub_models import RegionModel, RowsModel
from .converters import Rows2Regions

 


class Pipeline:
    def __init__(self, pred, model, dataset_name2id, model_id2name):
        self.pred = pred
        self.model = model
        
        self.rows_model = RowsModel()
        self.region_model = RegionModel()
        self.rows2regions = Rows2Regions({
            'model':model, 
            'tokenizer': pred.tokenizer,
            'is_merge_extract': True,
            'classes': model_id2name
        })
        self.name2id = dataset_name2id
        
        

    def __call__(self, path):
        pdf_json, pdf_img = self.pred.get_json_and_img(path)
        row_json = pdf_json['rows']
        self.rows_model.from_dict({"rows": row_json})
        self.rows2regions.convert(self.rows_model, self.region_model, pdf_img)
        pred_regions = self.region_model.to_dict()['regions']
        def name2id(r):
            r['label'] = self.name2id[r['label']]
            return r
            
        filtered_preds = [name2id(r) for r in pred_regions]
        return {
            "regions": filtered_preds
        }