BASE_PARAMS = {
        "node_block": {
            "linear_pred": [
                {"in": 15, "out": 256, "activation": "gelu"},
                {"in": 256, "out": 128, "activation": "gelu"},
            ],
            "gnn": [
                ("tag", {"batch_norm": True, "concat": True,
                         "in":128,  "in_gnn":256, "out_gnn":128,
                         "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                ("tag", {"batch_norm": False,"concat": True,
                         "in":128+128, "in_gnn":256, "out_gnn":128,
                         "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                    ],
        },
        "node_classifier_block": {
            "linear_post": [
                {"in": 128+128+128, "out": 256, "activation": "gelu"},
                {"in": 256, "out": 128, "activation": "gelu"},
                {"in": 128, "out": 6, "activation": "none"},
            ]
        },
        "post_node_block": {
            "linear_pred": [
                {"in": 128+128+128, "out": 256, "activation": "gelu"},
                {"in": 256, "out": 256, "activation": "gelu"},
                {"in": 256, "out": 128, "activation": "gelu"},
            ],
            "gnn": [
                ("tag", {"batch_norm": True, "concat": True,
                         "in":128,  "in_gnn":256, "out_gnn":128,
                         "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                ("tag", {"batch_norm": False,"concat": True,
                         "in":128+128, "in_gnn":256, "out_gnn":128,
                         "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                    ],
            "linear_post": [
                {"in": 128+128+128, "out": 256, "activation": "gelu"},
                {"in": 256, "out": 128, "activation": "gelu"},
                {"in": 128, "out": 128, "activation": "gelu"},
            ]
        },
        "edge_classifier_block": {
             "linear_post": [
                {"in": 2*(128+15)+4, "out": 256, "activation": "gelu"},
                {"in": 256, "out": 64, "activation": "gelu"},
                {"in": 64, "out": 1, "activation": "none"},
            ]
        },
        "save_frequency": 10,
        "epochs" : EPOCHS,
        "batch_size"  : 128,
        "learning_rate" : 0.001,
        "seg_k"  : 0.5,
        "loss_params" : {
            "edge_coef": 0.8,
            "node_coef": 0.2
        }
    }


import sys
import os
from pathlib import Path
import torch
import copy
import numpy as np
import json
from dotenv import load_dotenv

sys.path.append(os.path.join('..', '..'))
env_file = os.path.join('..', '..', '.env')
load_dotenv(env_file)

EPOCHS = int(os.environ.get('EPOCHS', '30'))

ds_path = Path('/home/daniil/disk01_1TB/datasets/')
# Кеш----------------------------------------------------------------------------
# doclaynet_cash_pdf_path = os.environ['DOCLAYNET_CASH_PDF_PATH']
# publaynet_cash_pdf_path = os.environ['PUBLAYNET_CASH_PDF_PATH']
doclaynet_cash_pdf_path = ds_path/'tmp/cache_miner/'
publaynet_cash_pdf_path = ds_path/'tmp/cache_miner_publaynet/'
# -------------------------------------------------------------------------------

# Тренеровка --------------------------------------------------------------------
# doclaynet_train_pdf_path = os.environ['DOCLAYNET_TRAIN_PDF_PATH']
# doclaynet_train_coco_path = os.environ['DOCLAYNET_TRAIN_COCO_PATH']

# publaynet_train_pdf_path = os.environ['PUBLAYNET_TRAIN_PDF_PATH']
# publaynet_train_coco_path = os.environ['PUBLAYNET_TRAIN_COCO_PATH']

doclaynet_train_pdf_path = ds_path/'DocLayNet_core_10k/PDF/'
doclaynet_train_coco_path = ds_path/'DocLayNet_core_10k/train.json'

publaynet_train_pdf_path = ds_path/'micro_publaynet_10k/pdfs/train/'
publaynet_train_coco_path = ds_path/'micro_publaynet_10k/publaynet/train.json'
# -------------------------------------------------------------------------------

# Тестирование ------------------------------------------------------------------
# doclaynet_test_pdf_path = os.environ['DOCLAYNET_TEST_PDF_PATH']
# doclaynet_test_coco_path = os.environ['DOCLAYNET_TEST_COCO_PATH']

# publaynet_test_pdf_path = os.environ['PUBLAYNET_TEST_PDF_PATH']
# publaynet_test_coco_path = os.environ['PUBLAYNET_TEST_COCO_PATH']
doclaynet_test_pdf_path = ds_path/'DocLayNet_core_mini/PDF/'
doclaynet_test_coco_path = ds_path/'DocLayNet_core_mini/train.json'

publaynet_test_pdf_path = ds_path/'micro_publaynet_10k/pdfs/dev/'
publaynet_test_coco_path = ds_path/'micro_publaynet_10k/publaynet/val.json'
# -------------------------------------------------------------------------------


from utils.experimenter import Experimenter 
from rows2regionsGLAM.utils.pdf_manager import PDFManager
from rows2regionsGLAM.utils.loger import Loger
from rows2regionsGLAM.utils.row_manager import RowManager
from rows2regionsGLAM.utils.ploter import Ploter
from rows2regionsGLAM.utils.trainer import Trainer
from rows2regionsGLAM.utils.tester import Tester
from rows2regionsGLAM.utils.cacher import Cacher
from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.pipeline import Pipeline
from rows2regionsGLAM.utils.coco_manager import COCOManager
from rows2regionsGLAM.utils.imbalance import calculate_imbalance
from rows2regionsGLAM.utils.tester import collect_maps, print_map_table
from rows2regionsGLAM.tokenizers import RowGLAMTokenizer
from rows2regionsGLAM.models import get_loss, get_model, get_tmp_params, save_model

from rows2regionsGLAM.datasetloaders.base_line_dataset import GLAMDataset


exp_path = 'result'
exp = Experimenter(name='test', result_save_path=exp_path)
tokenizer = RowGLAMTokenizer()
loger = Loger()

coco_manager_train_pub = COCOManager(loger=loger, coco_path=publaynet_train_coco_path, name_dataset='publaynet')
coco_manager_train_doc = COCOManager(loger=loger, coco_path=doclaynet_train_coco_path, name_dataset='doclaynet')
coco_manager_test_pub  = COCOManager(loger=loger, coco_path=publaynet_test_coco_path,  name_dataset='publaynet')
coco_manager_test_doc  = COCOManager(loger=loger, coco_path=doclaynet_test_coco_path,  name_dataset='doclaynet')

pred = PredProcessor(loger=loger, tokenizer=tokenizer)


dict_params = {
    f'train_{train_ds}_test_{test_ds}': {
        "model_param":   {"train_ds": train_ds, 'test_ds': test_ds},
        "dataset_param": {"train_ds": train_ds, 'test_ds': test_ds},
        "train_param":   {"train_ds": train_ds, 'test_ds': test_ds},
        "test_param":    {"train_ds": train_ds, 'test_ds': test_ds}
    }
    for train_ds in ['doc','pub'] for test_ds in [ 'doc', 'pub']
}
def fun_get_dataset_with_param(param):  
    
    train_dataset = GLAMDataset(
        coco_manager=coco_manager_train_pub if param['train_ds'] == 'pub' else coco_manager_train_doc, 
        cache_dir=publaynet_cash_pdf_path if param['train_ds'] == 'pub' else doclaynet_cash_pdf_path, 
        pdf_dir=publaynet_train_pdf_path if param['train_ds'] == 'pub' else doclaynet_train_pdf_path,
        default_index=0, 
        pred=pred,
        loger=loger,
    )
    
    test_dataset = GLAMDataset(
        coco_manager=coco_manager_test_pub if param['test_ds'] == 'pub' else coco_manager_test_doc, 
        cache_dir=publaynet_cash_pdf_path if param['train_ds'] == 'pub' else doclaynet_cash_pdf_path, 
        pdf_dir=publaynet_test_pdf_path if param['test_ds'] == 'pub' else doclaynet_test_pdf_path,
        default_index=0, 
        pred=pred,
        loger=loger,    
    )
    train_dataset.train()
    test_dataset.train()
    train_dataset.init()
    test_dataset.init()
    return {
        "train": train_dataset,
        "test": test_dataset
    }
        

def fun_get_model_with_param(param):
    train_ds = param["train_ds"]
    model_name = str(Path(exp_path, f'row2region_GLAM_{train_ds}'))
    
    model_params = BASE_PARAMS.copy()
    if train_ds == 'pub':
        model_params['node_classifier_block']['linear_post'][-1]['out'] = 6
    elif train_ds == 'doc':
        model_params['node_classifier_block']['linear_post'][-1]['out'] = 12
    else:
        raise Exception('неверная конфигурация')

    return {
         'model_name': model_name,
         'model_params': model_params
    }

def fun_train_model_with_param(model, dataset, param):

    model_name = model['model_name']
    model_params = model['model_params']
    
    dataset = dataset['train']
    dataset.train()
    node_imbalance, edge_imbalance = calculate_imbalance(dataset)
    model_params['loss_params']['node_imbalance'] = node_imbalance
    model_params['loss_params']['edge_imbalance'] = edge_imbalance

    if not Path(model_name).exists():
        model, num_restart = get_model(model_params, model_name)
        model_params['restart_num'] =  num_restart
        loss = get_loss(model_params['loss_params'])
        trainer = Trainer(model=model, dataset=dataset, loss=loss, train_param=model_params, loger=loger)
        trainer.start_train()
        model = trainer.model
        save_model(model, model_name)
        
def fun_test_model_with_param(model, dataset, param):
    test_dataset = dataset['test']
    train_dataset = dataset['train']
    model_name = model['model_name']
    
    train_ds=param["train_ds"]
    test_ds=param["test_ds"]
    if train_ds == 'pub' and test_ds == 'pub':
        model_id2name = train_dataset.coco_manager.classes
        dataset_name2id = {
            'other': 0, 'text': 1, 'title': 2, 'list': 3, 'table': 4, 'figure': 5
        }
    elif train_ds == 'doc' and test_ds == 'doc':
        model_id2name = train_dataset.coco_manager.classes
        dataset_name2id = {
            'other': 0,
            'Caption': 1,
            'Footnote': 2,
            'Formula': 3,
            'List-item': 4,
            'Page-footer': 5,
            'Page-header': 6,
            'Picture': 7,
            'Section-header': 8,
            'Table': 9,
            'Text': 10,
            'Title': 11
        }
    elif train_ds == 'doc' and test_ds == 'pub':
        model_id2name = {0: 'other', 1: 'text', 2: 'text', 3: 'other', 4: 'list', 5: 'other', 6:'other', 7:'figure', 8:'title', 9:'table', 10: 'text', 11:'title'}
        dataset_name2id = {
            'other': 0, 'text': 1, 'title': 2, 'list': 3, 'table': 4, 'figure': 5
        }
    elif train_ds == 'pub' and test_ds == 'doc':
        model_id2name = {0: 'other', 1:'Text', 2:'Section-header', 3:'List-item', 4: 'Table', 5: 'Picture'}
        dataset_name2id = {
            'other': 0,
            'Caption': 1,
            'Footnote': 2,
            'Formula': 3,
            'List-item': 4,
            'Page-footer': 5,
            'Page-header': 6,
            'Picture': 7,
            'Section-header': 8,
            'Table': 9,
            'Text': 10,
            'Title': 11
        }

    tmp_rez = Path(model_name+f'_on_{test_ds}_test.txt')
    if tmp_rez.exists():
        with open(tmp_rez, 'r') as f:
            rez = json.load(f)
        return rez
        
    model_params = model['model_params'].copy()
    model_params['node_classifier_block']['linear_post'][-1]['activation'] = "softmax"
    model_params['edge_classifier_block']['linear_post'][-1]['activation'] = "sigmoid"

    
    model, _ = get_model(model_params, model_name)
    pipeline = Pipeline(pred=pred, model=model, dataset_name2id=dataset_name2id, model_id2name=model_id2name)
    tester = Tester(pipeline=pipeline, dataset=test_dataset, loger=loger)
    tester.calculate()
    grid_cls, map_cls = tester.get_results()
    rez = {**grid_cls, **map_cls}
    with open(tmp_rez, 'w') as f:
        json.dump(rez, f)
    return rez
    

def fun_result_to_row(train_result, test_result):
    return test_result


exp.experiment(
    fun_get_model_with_param, 
    fun_get_dataset_with_param,
    fun_train_model_with_param,
    fun_test_model_with_param,
    fun_result_to_row,
    dict_params
)