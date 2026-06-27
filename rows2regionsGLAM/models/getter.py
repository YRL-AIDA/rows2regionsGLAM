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
    base = os.path.basename(path_model)
    name, ext = os.path.splitext(base)

    if restart_num is None and os.path.exists(path_model):
        model.load_state_dict(torch.load(path_model, weights_only=True))
        return None

    best_path = path_model + "_best"
    if restart_num is None and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
        return None

    pattern = f"{name}_tmp_"
    tmp_files = [f for f in os.listdir(dir_model) if f.startswith(pattern) and f.endswith(ext)]
    if not tmp_files:
        return None

    nums = []
    for f in tmp_files:
        without_ext = f[:-len(ext)] if ext else f
        try:
            num = int(without_ext.split("_tmp_")[-1])
            nums.append(num)
        except ValueError:
            continue

    if not nums:
        return None

    if restart_num is None:
        restart_num = max(nums)
    elif restart_num not in nums:
        restart_num = max(nums)

    checkpoint_path = os.path.join(dir_model, f"{name}_tmp_{restart_num}{ext}")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        return restart_num
    return None