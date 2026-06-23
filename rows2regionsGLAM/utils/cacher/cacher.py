from pathlib import Path
import json
from json import JSONEncoder
import os
import torch
from torch.utils.data import Dataset
import warnings

class EncodeTensor(JSONEncoder, Dataset):  
    def default(self, obj):
        if isinstance(obj, torch.Tensor):  
            return obj.cpu().detach().numpy().tolist()  
        return super(EncodeTensor, self).default(obj)  
    

class Cacher:
    def __init__(self, **conf):
        if "cache_dir" in conf.keys():
            self.cache_dir = Path(conf["cache_dir"])
        else:
            raise Exception('Укажите папку для cache ("cache_dir": path)')
        
        if not self.cache_dir.exists():
            self.cache_dir.mkdir()
        elif len(list(self.cache_dir.iterdir())) != 0:
            warnings.warn("Кеш не пустой !!!", DeprecationWarning)
            
        if "cache_fun" in conf.keys():
            self.cache_fun = conf["cache_fun"]
        else:  
            raise Exception('Укажите функцию для кеширования ("cache_fun": function (unic_name)) -> json')
    
    def __call__(self, unic_name):
        path_file = self.cache_dir / f"{unic_name}.json"
        if path_file.exists():
            with open(path_file, "r") as f:
                res = json.load(f)
            return res
        
        data = self.cache_fun(unic_name)
        tmp = path_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, cls=EncodeTensor)
        os.rename(tmp, path_file)
        return data