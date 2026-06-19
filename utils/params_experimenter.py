import warnings
import sys, os
import torch
from pathlib import Path
from dotenv import load_dotenv
import json

PATH_PROJECT = os.path.join('..')
sys.path.append(PATH_PROJECT)
env_file = os.path.join(PATH_PROJECT, '.env')
load_dotenv(env_file)
warnings.filterwarnings('ignore', message='Converting sparse tensor to CSR format')
warnings.filterwarnings('ignore', message='Implicit dimension choice for softmax')

from pager.page_model.sub_models import RegionModel, RowsModel

from .experimenter import Experimenter 
from rows2regionsGLAM.utils.loger import Loger
from rows2regionsGLAM.utils.coco_manager import COCOManager
from rows2regionsGLAM.utils.cacher import Cacher
from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.datasetloaders.base_line_dataset import GLAMDataset
from rows2regionsGLAM.utils.trainer import Trainer
from rows2regionsGLAM.utils.tester import Tester
from rows2regionsGLAM.utils.imbalance import calculate_imbalance
from rows2regionsGLAM.tokenizers import RowGLAMTokenizer
from rows2regionsGLAM.pipeline import Pipeline
from rows2regionsGLAM.models import get_loss, get_model, get_tmp_params, save_model

exp_path = 'result'
train_dataset_name = os.environ['NAME_DATASET']
train_dataset_path = os.environ['DATASET_PATH']
train_dataset_coco = os.environ['COCO_PATH']

test_dataset_name = os.environ['NAME_TEST_DATASET']
test_dataset_path = os.environ['TEST_PATH']
test_dataset_coco = os.environ['TEST_COCO_PATH']

cache_pdf = os.environ["CASH_PDF_PATH"]


exp = Experimenter(name='test', result_save_path=exp_path)

loger = Loger()
coco_manager_train = COCOManager(loger=loger, coco_path=train_dataset_coco, name_dataset=train_dataset_name)
coco_manager_test  = COCOManager(loger=loger, coco_path=test_dataset_coco,  name_dataset=test_dataset_name)
tokenizer = RowGLAMTokenizer()
pred = PredProcessor(loger=loger, tokenizer=tokenizer)



def fun_get_dataset_with_param(param):  
    print('*'*70 + 'dataset')
    train_dataset = GLAMDataset(coco_manager=coco_manager_train, default_index=0, pred=pred,
                          loger=loger, cache_dir=cache_pdf, pdf_dir=train_dataset_path)
    print('-'*80 + 'init train')
    train_dataset.train()# Для ускорения
    train_dataset.init()
    print('-'*80 + 'create test')
    test_dataset = GLAMDataset(coco_manager=coco_manager_test, default_index=0, pred=pred,
                               loger=loger, cache_dir=cache_pdf, pdf_dir=test_dataset_path)
    print('-'*80 + 'init test')
    test_dataset.train() # Для ускорения
    test_dataset.init()
    return {
        "train": train_dataset,
        "test": test_dataset
    }
        

def fun_get_model_with_param(param):
    print('*'*70 + 'model')
    name=param['name']
    model_params=param
    model_name = str(Path(exp_path, f'row2region_GLAM_{name}'))
    return {
         'model_name': model_name,
         'model_params': model_params
    }

def fun_train_model_with_param(model, dataset, param):
    print('*'*70 + 'train')
    model_name = model['model_name']
    model_params = model['model_params'].copy()
    
    model_params['node_classifier_block']['linear_post'][-1]['activation'] = "none"
    model_params['edge_classifier_block']['linear_post'][-1]['activation'] = "none"

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
    print('*'*70 + 'test')
    test_dataset = dataset['test']
    train_dataset = dataset['train']
    model_name = model['model_name']
    tmp_rez = Path(model_name+'_res.txt')
    if tmp_rez.exists():
        with open(tmp_rez, 'r') as f:
            rez = json.load(f)
        return rez
        
    model_params = model['model_params'].copy()
    model_params['node_classifier_block']['linear_post'][-1]['activation'] = "softmax"
    model_params['edge_classifier_block']['linear_post'][-1]['activation'] = "sigmoid"
    
    
    model, _ = get_model( model_params, model_name)

    model_id2name = train_dataset.coco_manager.classes
    dataset_name2id = {val:key for key, val in model_id2name.items()}
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



def start_experiments(dict_model_params):
    """
    name_exp: params_model <- есть имя эксперимента
    model_type: main
    """
    
    dict_params = {
        name_exp: {
            "model_param": {**params, "name":name_exp},
            "dataset_param": {},
            "train_param": params,
            "test_param": params
        }
    
        for name_exp, params in dict_model_params.items()
    }
    exp.experiment(
        fun_get_model_with_param, 
        fun_get_dataset_with_param,
        fun_train_model_with_param,
        fun_test_model_with_param,
        fun_result_to_row,
        dict_params
    )