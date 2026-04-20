from .rowGLAM_base import TorchModelBase, PARAMS_BASE, CustomLossBase
from .rowGLAM_custom import TorchModel, PARAMS, CustomLoss
import torch


def get_tmp_params(type_model):
    if type_model == "base":
        return PARAMS_BASE
    elif type_model == "custom":
        return PARAMS
    else:
        raise ValueError("type_model error")

def get_model(type_model, params):
    if type_model == "base":
        return TorchModelBase(params)
    elif type_model == "custom":
        return TorchModel(params)
    else:
        raise ValueError("type_model error")

def get_loss(type_model, params):
    if type_model == "base":
        return CustomLossBase(params)
    elif type_model == "custom":
        return CustomLoss(params)
    else:
        raise ValueError("type_model error")