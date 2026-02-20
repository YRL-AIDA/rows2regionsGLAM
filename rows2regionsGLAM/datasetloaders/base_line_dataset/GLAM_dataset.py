from torch.utils.data import Dataset 
import torch
import json
import numpy as np
import os 
from ...utils.coco_manager import COCOManager
from json import JSONEncoder 
import warnings
from collections import defaultdict

def cache_exists(cache_path):
    """Проверяет, существует ли кэшированный файл."""
    return os.path.exists(cache_path)

def load_cache(cache_path):
    """Загружает кэшированные данные."""
    with open(cache_path, 'r') as f:
        return json.load(f)

class GLAMDataset(Dataset):
    def __init__(self, conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        self.loger.time_log()
        self.loger("Create Dataset")

        if "pdf_dir" in conf.keys(): 
            self.pdf_dir  = conf["pdf_dir"] 
        else:
            raise Exception('Укажите папку до pdf файлов ("pdf_dir": path)')
        self.loger(f"Path Dataset: {self.pdf_dir}")

        if "coco_file" in conf.keys(): 
            self.coco_file  = conf["coco_file"] 
            self.coco_manager = COCOManager({"loger": self.loger, "coco_path": self.coco_file})
        else:
            raise Exception('Укажите папку до COCO-разметки файлов ("coco_file": path)')
        self.loger(f"Path COCO: {self.coco_file}")

        if "count_class" in conf.keys(): 
            self.count_class = conf["count_class"] 
        else:
            raise Exception('Укажите число классов в наборе ("count_class": int)')

        if "name_dataset" in conf.keys():
            self.name_dataset = conf["name_dataset"]
        else:
            raise Exception('Укажите название набора данных ("name_dataset" : str)')

        if "default_index" in conf.keys(): 
            self.default_index = conf["default_index"] 
        else:
            raise Exception('Укажите индекс класса по умолчанию ("default_index": int)')

        if "cache_dir" in conf.keys():
            self.cache_dir = conf["cache_dir"]
            if os.path.exists(self.cache_dir):
                if len(os.listdir(self.cache_dir)) != 0:
                    warnings.warn("Кеш не пустой !!!", DeprecationWarning)
            else:
                os.mkdir(self.cache_dir)
        else:
            raise Exception('Укажите папку для cache ("cache_dir": path)')
        
        if "pdf2torch_dict" in conf.keys(): 
            self.pdf2torch_dict = conf["pdf2torch_dict"] 
        else:
            raise Exception('Напишите функцию перевода pdf в torch_dict ("pdf2torch_dict": pdf2torch_dict(path_pdf, coco_dict_file) )')

        if "to_ROM" in conf.keys():
            self.to_ROM = conf["to_ROM"]
        else:
            raise Exception('Укажите сохранять ли датасет в оперативной памяти ("to_ROM": bool)')

        self.device = torch.device(conf['device'] if 'device' in conf else 'cpu')

        pdfs = [f for f in os.listdir(self.pdf_dir) if f.split('.')[-1] == 'pdf' and not os.path.isdir(f)]
        jsons = [f for f in os.listdir(self.cache_dir)]
        pdfs.sort()
        self.count = len(pdfs)
        self.pdf_names = [os.path.basename(pdf) for pdf in pdfs]
        self.cache_names = [os.path.basename(js) for js in jsons]
        self.coco_ann = self.coco_manager.get_regions_from_json()[0]

        if self.to_ROM:
            self.memory = dict()
            self.load_to_memory()

    def test_cache(self): 
        files  = sorted(os.listdir(self.cache_dir))

        if os.path.exists("error_list_file.txt"):
            with open("error_list_file.txt", "r") as f:
                lines = f.readlines()
            error_file = [int(line.split(" ")[0]) for line in lines]
        else:       
            print("TEST OPEN FILE:")
            json_error = []
            key_error = []
            N = len(files)
            for i, file in enumerate(files):
                print(f"{i+1}/{N} ({(i+1)/N*100:.2f} %)" + " "*10, end="\r")
                try:
                    path = os.path.join(self.cache_dir, file)
                    with open(path, "r") as f: 
                        j = json.load(f)
                    for k in ["inds", "X", "Y", "N", "true_edges", "true_nodes"]:                    
                        if not k in j:
                            key_error.append(i)
                            raise KeyError(f"{k} not in {file}")
                    if len(j['X']) <= 1 and len(j['Y']) <= 1:
                        raise Exception('one node or one edge')
                except:
                    json_error.append(i)
            
            if len(key_error) != 0:
                print("KEY ERROR FILES:")
                self.loger("KEY ERROR FILES:")
                for i in key_error:
                    print(files[i])
                    self.loger(files[i])

            if len(json_error) != 0:
                print("JSON ERROR FILES:")
                self.loger("JSON ERROR FILES:")
                for i in json_error:
                    print(files[i])
                    self.loger(files[i])
            error_file = sorted(key_error + json_error, reverse=True)
            with open("error_list_file.txt", "w") as f:
                for i in error_file:
                    f.write(str(i) + " "+ files[i] + '\n')
        for i in error_file:
            del files[i]
        self.files = files
        self.count = len(self.files)

    def load_to_memory(self):
        for i, name_file in enumerate(self.pdf_names):
            if not name_file+'.json' in self.cache_names:
                try:
                    data = self.cache_file(name_file)
                except:
                    # self.loger(f"ERROR file: {name_file}")
                    return {}
            else:
                path = os.path.join(self.cache_dir, name_file+'.json')

                with open(path, 'r') as f:
                    data = json.load(f)

            self.memory[i] = self.get_torch_data(data)


    def get_torch_data(self, data_input):
        data = {}
        if len(data_input.keys()) == 0:
            return {}
        data['X'] = torch.tensor(data_input['X'], dtype=torch.float32).to(self.device)
        data['Y'] = torch.tensor(data_input['Y'], dtype=torch.float32).to(self.device)
        N = data_input["N"]
        i = data_input['inds']
        data["N"] = N
        data['inds'] = i
        index_for_mtrx = [i[0]+i[1], i[1]+i[0]]
        sp_A = torch.sparse_coo_tensor(indices=index_for_mtrx, values=[1 for e in index_for_mtrx[0]], size=(N, N), dtype=torch.float32).to(self.device)
        data['sp_A'] = sp_A
        data['true_edges'] = torch.tensor([0 if i is None else i for i in data_input['true_edges']], dtype=torch.float32).to(self.device)
        data['true_nodes'] = self.__class_to_vec(data_input['true_nodes']).to(self.device)
        # data['file_name'] = data_input['file_name']  #TODO: Вернуть после эксп
        return data
        

    def __len__(self):
        return self.count

    def __class_to_vec(self, classes):
        def vec_class(c):
            base_vec = [0 for _ in range(self.count_class)]
            
            if c is None:
                base_vec[self.default_index] = 1
            else:
                base_vec[c] = 1
            return base_vec
        return torch.tensor([vec_class(c) for c in classes], dtype=torch.float32)

    def __getitem__(self, idx):
        if idx >= self.count:
            raise IndexError()  
        name_file = self.pdf_names[idx]
        if self.to_ROM:    
            data = self.memory[idx] if idx >= 0 else self.memory[self.count+idx]
            data['file_name'] = '.'.join(name_file.split('.')[:-1])  #TODO: Убрать после эксп
            return data


        if not name_file+'.json' in self.cache_names:
            try:
                data = self.cache_file(name_file)
            except:
                # self.loger(f"ERROR file: {name_file}")
                return {}
        else:
            path = os.path.join(self.cache_dir, name_file+'.json')

            with open(path, 'r') as f:
                data = json.load(f)
        data['file_name'] ='.'.join(name_file.split('.')[:-1]) #TODO: Убрать после эксп
        return self.get_torch_data(data) 

    def cache_file(self, name_file):
        name_json = os.path.join(self.cache_dir, name_file + '.json')

        if cache_exists(name_json):
            return load_cache(name_json)

        path_file = os.path.join(self.pdf_dir, name_file)
        json_res = self.pdf2torch_dict(path_file, self.coco_ann[name_file], self.name_dataset)
        json_res['file_name'] = name_file 
        with open(name_json, 'w') as f:
            json.dump(json_res, f, cls=EncodeTensor)
        self.cache_names.append(os.path.basename(name_json))
        return json_res

    def __str__(self):
        return f"""
            DATASET INFO:
            count row: {len(self)}
            first: {self[0].keys()}
            \t A:{np.shape(self[0]["sp_A"])}
            \t nodes_feature:{np.shape(self[0]["X"])}
            \t edges_feature:{np.shape(self[0]["Y"])}
            \t true_edges:{np.shape(self[0]["true_edges"])}
            end:{self[-1].keys()}
            \t A:{np.shape(self[-1]["sp_A"])}
            \t nodes_feature:{np.shape(self[-1]["X"])}
            \t edges_feature:{np.shape(self[-1]["Y"])}
            \t true_edges:{np.shape(self[-1]["true_edges"])}
        """
    


class EncodeTensor(JSONEncoder, Dataset):  
    def default(self, obj):
        if isinstance(obj, torch.Tensor):  
            return obj.cpu().detach().numpy().tolist()  
        return super(EncodeTensor, self).default(obj)  