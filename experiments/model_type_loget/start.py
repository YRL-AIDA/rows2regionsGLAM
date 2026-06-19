import warnings
import sys, os
import torch
from pathlib import Path
from dotenv import load_dotenv

PATH_PROJECT = os.path.join('..', '..')
sys.path.append(PATH_PROJECT)
env_file = os.path.join(PATH_PROJECT, '.env')
load_dotenv(env_file)
warnings.filterwarnings('ignore', message='Converting sparse tensor to CSR format')
warnings.filterwarnings('ignore', message='Implicit dimension choice for softmax')

from pager.page_model.sub_models import RegionModel, RowsModel

from utils.experimenter import Experimenter 
from rows2regionsGLAM.utils.loger import Loger
from rows2regionsGLAM.utils.coco_manager import COCOManager
from rows2regionsGLAM.utils.cacher import Cacher
from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.datasetloaders.base_line_dataset import GLAMDataset
from rows2regionsGLAM.utils.trainer import Trainer
from rows2regionsGLAM.utils.tester import Tester
from rows2regionsGLAM.models import get_loss, get_model, get_tmp_params
from rows2regionsGLAM.utils.imbalance import calculate_imbalance
from rows2regionsGLAM.converters import Rows2Regions
from rows2regionsGLAM.tokenizers import RowGLAMTokenizer

EPOCHS = int(os.environ.get('EPOCHS', '30'))

def fun_get_dataset_with_param(param):  
    train_dataset = GLAMDataset(coco_manager=coco_manager_train, default_index=0, pred=pred_train,
                          loger=loger, cache_dir=cache_pdf, pdf_dir=train_dataset_path)
    # Создание Cache
    N = len(train_dataset)
    for i, d in enumerate(train_dataset):
        print(f"{(i+1)/N*100:4.2f} %", end='\r')

    test_dataset = GLAMDataset(coco_manager=coco_manager_test, default_index=0, pred=pred_test,
                               loger=loger, cache_dir=cache_pdf, pdf_dir=test_dataset_path)
    # Создание Cache
    N = len(test_dataset)
    for i, d in enumerate(test_dataset):
        print(f"{(i+1)/N*100:4.2f} %", end='\r')

    return {
        "train": train_dataset,
        "test": test_dataset
    }
        

def fun_get_model_with_param(param):
    type_model=param["type"]

    
    model_name = str(Path(exp_path, f'row2region_GLAM_{type_model}'))
    
    if type_model == "loget":
        model_params = {
            "node_block": { # Первый слой содержит число features
                "linear_pred": [
                    {"in": 15, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                ],
                "gnn": [
                    ("tag", {"batch_norm": True, "concat": True,
                             "in":128,  "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
                    ("tag", {"batch_norm": False,"concat": True,
                             "in":128+128, "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                        ],
                # "linear_post": [ # PRED похож на POST
                #     {"in": 128+128+128, "out": 256, "activation": "gelu"},
                #     {"in": 256, "out": 128, "activation": "gelu"},
                #     {"in": 128, "out": 32, "activation": "gelu"},
                # ]
            },
            "node_classifier_block": {
                "linear_post": [ # Удобно задать MLP как только POST часть
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
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
                    ("tag", {"batch_norm": False,"concat": True,
                             "in":128+128, "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                        ],
                "linear_post": [ # PRED похож на POST
                    {"in": 128+128+128, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                    {"in": 128, "out": 128, "activation": "gelu"},
                ]
            }, # НЕ ОБЯЗАТЕЛЬНЫЙ
            # "conjugate_edge_block": {}, # НЕ ОБЯЗАТЕЛЬНЫЙ
            "edge_classifier_block": {
                 "linear_post": [ # Удобно задать MLP как только POST часть
                    {"in": 2*(128+15)+4, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 64, "activation": "gelu"},
                    {"in": 64, "out": 1, "activation": "none"},
                ]
            }
        }
    elif type_model=="loget_no":
        model_params = {
            "node_block": { # Первый слой содержит число features
                "linear_pred": [
                    {"in": 15, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                ],
                "gnn": [
                    ("tag", {"batch_norm": True, "concat": True,
                             "in":128,  "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
                    ("tag", {"batch_norm": False,"concat": True,
                             "in":128+128, "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                        ],
                # "linear_post": [ # PRED похож на POST
                #     {"in": 128+128+128, "out": 256, "activation": "gelu"},
                #     {"in": 256, "out": 128, "activation": "gelu"},
                #     {"in": 128, "out": 32, "activation": "gelu"},
                # ]
            },
            "node_classifier_block": {
                "linear_post": [ # Удобно задать MLP как только POST часть
                    {"in": 128+128+128, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                    {"in": 128, "out": 6, "activation": "softmax"},
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
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}), # Еще есть concat с прошлым слоем
                    ("tag", {"batch_norm": False,"concat": True,
                             "in":128+128, "in_gnn":256, "out_gnn":128, 
                             "gnn_activation":"gelu", "activation":"gelu", "K":3}),
                        ],
                "linear_post": [ # PRED похож на POST
                    {"in": 128+128+128, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 128, "activation": "gelu"},
                    {"in": 128, "out": 128, "activation": "gelu"},
                ]
            }, # НЕ ОБЯЗАТЕЛЬНЫЙ
            # "conjugate_edge_block": {}, # НЕ ОБЯЗАТЕЛЬНЫЙ
            "edge_classifier_block": {
                 "linear_post": [ # Удобно задать MLP как только POST часть
                    {"in": 2*(128+15)+4, "out": 256, "activation": "gelu"},
                    {"in": 256, "out": 64, "activation": "gelu"},
                    {"in": 64, "out": 1, "activation": "none"},
                ]
            }
        }
    else:
        raise Exception('неверная конфигурация')

    model_params["epochs"] = EPOCHS
    model_params["batch_size"] = 64
    model_params["learning_rate"]=0.001
    model_params["seg_k"] = 0.5,
    model_params["loss_params"] = {
                "edge_coef": 0.8,
                "node_coef": 0.2}
    return {
         'model_name': model_name,
         'model_params': model_params
    }

def fun_train_model_with_param(model, dataset, param):
    
    model_name = model['model_name']
    model_params = model['model_params']

    dataset = dataset['train']
    type_model=param["type"]

    
    publaynet_imbalance, edge_imbalance = calculate_imbalance(dataset)
    model_params['loss_params']['publaynet_imbalance'] = publaynet_imbalance
    model_params['loss_params']['edge_imbalance'] = edge_imbalance
    
    if not Path(model_name).exists():
        trainer_pub = Trainer(conf={"loger": loger, "params": model_params, "model_name": model_name})
        trainer_pub.start_train(5, dataset, type_model='main')
    
def fun_test_model_with_param(model, dataset, param):
    test_dataset = dataset['test']
    type_model = param["type"]
    model_name = model['model_name']
    model_params = model['model_params']

    
    model_params['node_classifier_block']['linear_post'][-1]['activation'] = "softmax"
    model_params['edge_classifier_block']['linear_post'][-1]['activation'] = "sigmoid"
    
    model = get_model('main', model_params)
    model.load_state_dict(torch.load(model_name, weights_only=True))
    
    rows_model = RowsModel()
    region_model = RegionModel()
    rows2regions = Rows2Regions({
        'model':model, 
        'tokenizer': tokenizer,
        'is_merge_extract': True,
        'classes': coco_manager_test.classes
    })
    tester = Tester(loger=loger,
                    pred=test_dataset.pred,
                    rows_model=rows_model,
                    rows2regions=rows2regions,
                    region_model=region_model)

    metrics = tester.calculate_target_and_preds(test_dataset)

    grid_cls, map_cls = tester.get_results(metrics)
    
    return {**grid_cls, **map_cls}

def fun_result_to_row(train_result, test_result):
    return test_result

if __name__ == "__main__":
    train_dataset_name = os.environ['NAME_DATASET']
    train_dataset_path = os.environ['DATASET_PATH']
    train_dataset_coco = os.environ['COCO_PATH']
    
    test_dataset_name = os.environ['NAME_TEST_DATASET']
    test_dataset_path = os.environ['TEST_PATH']
    test_dataset_coco = os.environ['TEST_COCO_PATH']

    cache_pdf = os.environ["CASH_PDF_PATH"]
    
    exp_path = 'test_path'
    exp = Experimenter(name='test', result_save_path=exp_path)

    loger = Loger()
    coco_manager_train = COCOManager(loger=loger, coco_path=train_dataset_coco, name_dataset=train_dataset_name)
    coco_manager_test  = COCOManager(loger=loger, coco_path=test_dataset_coco,  name_dataset=test_dataset_name)
    tokenizer = RowGLAMTokenizer()
    pred_train = PredProcessor(loger=loger, coco_manager=coco_manager_train, tokenizer=tokenizer)
    pred_test =  PredProcessor(loger=loger, coco_manager=coco_manager_test,  tokenizer=tokenizer)
    
    dict_params = {
        name_model: {
            "model_param": {"type": name_model},
            "dataset_param": {},
            "train_param": {"type": name_model},
            "test_param": {"type": name_model}
        }
    
        for name_model in ['loget',  'loget_no'] # 'base',
    }
    exp.experiment(
        fun_get_model_with_param, 
        fun_get_dataset_with_param,
        fun_train_model_with_param,
        fun_test_model_with_param,
        fun_result_to_row,
        dict_params
    )