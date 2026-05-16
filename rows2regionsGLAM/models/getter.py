from .rowGLAM import TorchModel, PARAMS, CustomLoss 
import torch
import os

def get_tmp_params():
    return PARAMS

def get_model(params, path_model): 
    name_model = os.path.basename(path_model)
    model = TorchModel(params)
    model.path = path_model
    restart_num = _load_checkpoint(model, path_model)
    return model, restart_num

def get_loss(params):
    return CustomLoss(params)

def save_model(model, path_model):
    torch.save(model.state_dict(), path_model)

def _load_checkpoint(model, path_model, restart_num=None):
    dir_model = os.path.dirname(path_model)
    name_model = os.path.basename(path_model)
    names = [n for n in os.listdir(dir_model) if name_model+'_tmp_' in n]
    if restart_num is None:
        list_num = [int(n.split("_tmp_")[-1]) for n in names]
        if len(list_num) == 0:
            return
        restart_num = max(list_num) 

    checkpoint_path = os.path.join(dir_model, name_model+f"_tmp_{restart_num}")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return restart_num