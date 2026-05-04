from torch.utils.data import Dataset 
import torch
import json
import numpy as np
import os 
from ...utils.coco_manager import COCOManager
from ...utils.cacher import Cacher
from pathlib import Path

class GLAMDataset(Dataset):
    def __init__(self, **conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        self.loger.time_log()
        self.loger("Create Dataset")

        if "pdf_dir" in conf.keys(): 
            self.pdf_dir  = Path(conf["pdf_dir"])
        else:
            raise Exception('Укажите папку до pdf файлов ("pdf_dir": path)')
        self.loger(f"Path Dataset: {self.pdf_dir}")

        # if "coco_file" in conf.keys(): 
        #     self.coco_file  = conf["coco_file"] 
        #     self.coco_manager = COCOManager({"loger": self.loger, "coco_path": self.coco_file})
        # else:
        #     raise Exception('Укажите папку до COCO-разметки файлов ("coco_file": path)')
        # self.loger(f"Path COCO: {self.coco_file}")

        if "coco_manager" in conf.keys(): 
            self.coco_manager = conf['coco_manager']
    
            self.count_class = len(self.coco_manager.classes)
        else:
            raise Exception('Укажите число классов в наборе ("count_class": int)')

        # if "name_dataset" in conf.keys():
        #     self.name_dataset = conf["name_dataset"]
        # else:
        #     raise Exception('Укажите название набора данных ("name_dataset" : str)')

        if "default_index" in conf.keys(): 
            self.default_index = conf["default_index"] 
        else:
            raise Exception('Укажите индекс класса по умолчанию ("default_index": int)')

        if "cache_dir" in conf.keys():
            self.cacher = Cacher(cache_dir=conf["cache_dir"], 
                                 cache_fun=self.pdf2json_for_model)
        else:
            raise Exception('Укажите папку для cache ("cache_dir": path)')
        
        if "pred" in conf.keys(): 
            self.pred = conf["pred"] 
        else:
            raise Exception('Напишите функцию перевода pdf в torch_dict ("pred": pred(path_pdf))')

        pdfs = [f for f in os.listdir(self.pdf_dir) if f.split('.')[-1] == 'pdf' and not os.path.isdir(f)]
        pdfs.sort()
        self.count = len(pdfs)
        self.pdf_names = [os.path.basename(pdf) for pdf in pdfs]

    def pdf2json_for_model(self, name_file):
        torch_dict = self.pred(self.pdf_dir/name_file)
        del torch_dict['sp_A']
        return torch_dict

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
        name_file = self.pdf_names[idx]
        try:
            data = self.cacher(name_file)
        except Exception as e:
            print(e)
            print("cache error", name_file)
            return {}
        if len(data.keys()) == 0:
            return {}
        try:
            data['X'] = torch.tensor(data['X'], dtype=torch.float32)
            data['Y'] = torch.tensor(data['Y'], dtype=torch.float32)
            N = data["N"]
            i = data['inds']
            index_for_mtrx = [i[0]+i[1], i[1]+i[0]]
            sp_A = torch.sparse_coo_tensor(indices=index_for_mtrx, values=[1 for e in index_for_mtrx[0]], size=(N, N), dtype=torch.float32)
            data['sp_A'] = sp_A
            data['true_edges'] = torch.tensor([0 if i is None else i for i in data['true_edges']], dtype=torch.float32)
            data['true_nodes'] = self.__class_to_vec(data['true_nodes'])
            data['file_name'] = '.'.join(name_file.split('.')[:-1])
        except:
            print(name_file)
            return {}
        return data

    

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