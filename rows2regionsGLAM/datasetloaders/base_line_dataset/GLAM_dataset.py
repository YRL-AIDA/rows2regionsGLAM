from torch.utils.data import Dataset 
import torch
torch.sparse.check_sparse_tensor_invariants.disable()
import json
import numpy as np
import os 
import random
import warnings
from ...utils.coco_manager import COCOManager
from ...utils.cacher import Cacher
from ...utils.intersect_util import get_num_regions_of_rows
from pathlib import Path
from dotenv import load_dotenv

from pagerlib.dtypes import ImageSegment

# env_file = os.path.join('..', '.env')
# load_dotenv(env_file)

class GLAMDataset(Dataset):
    def __init__(self, **conf):
        if "loger" not in conf.keys():
            self.loger = None
        else:
            self.loger = conf['loger']
        if self.loger:
            self.loger.time_log()
            self.loger("Create Dataset")

        if "pdf_dir" in conf.keys(): 
            self.pdf_dir  = Path(conf["pdf_dir"])
        else:
            raise Exception('Укажите папку до pdf файлов ("pdf_dir": path)')
        if self.loger:
            self.loger(f"Path Dataset: {self.pdf_dir}")
        
        if "coco_manager" in conf.keys(): 
            self.coco_manager = conf['coco_manager']
    
            self.count_class = len(self.coco_manager.classes)
        else:
            raise Exception('Укажите число классов в наборе ("count_class": int)')

        if "default_index" in conf.keys(): 
            self.default_index = conf["default_index"] 
        else:
            raise Exception('Укажите индекс класса по умолчанию ("default_index": int)')
        
        if "cache_dir" in conf.keys():
            self.cacher = Cacher(cache_dir=conf["cache_dir"], 
                                 cache_fun=self.pdf2json_for_model,
                                 warn_nonempty=conf.get("warn_cache_nonempty", False))
        else:
            raise Exception('Укажите папку для cache ("cache_dir": path)')
        
        if "pred" in conf.keys(): 
            self.pred = conf["pred"] 
        else:
            raise Exception('Напишите функцию перевода pdf в torch_dict ("pred": pred(path_pdf))')

        # ── Новые параметры ──
        self._cache_only = conf.get("cache_only", False)
        self._data_fraction = conf.get("data_fraction", 1.0)
        self._data_seed = conf.get("data_seed", None)
        self._ram_cache_flag = conf.get("ram_cache", False)
        self._ram_cache = None  # dict[str, dict] — заполняется при первом __getitem__

        self.device = torch.device(os.environ.get('DEVICE', 'cpu'))
        pdfs = [f for f in os.listdir(self.pdf_dir) if f.split('.')[-1] == 'pdf' and not os.path.isdir(f)]
        pdfs.sort()
        pdf_names = [os.path.basename(pdf) for pdf in pdfs]

        # Фаза 1: cache_only — фильтрация по наличию кеша
        if self._cache_only:
            cache_dir = Path(conf["cache_dir"])
            filtered = []
            skipped = []
            for name in pdf_names:
                if (cache_dir / f"{name}.json").exists():
                    filtered.append(name)
                else:
                    skipped.append(name)
            if skipped:
                msg = f"cache_only: пропущено {len(skipped)}/{len(pdf_names)} PDF без кеша"
                warnings.warn(msg)
                if self.loger:
                    self.loger(msg)
                    for s in skipped[:10]:
                        self.loger(f"  SKIP: {s}")
                    if len(skipped) > 10:
                        self.loger(f"  ... и ещё {len(skipped) - 10}")
            pdf_names = filtered

        # Фаза 2: data_fraction — случайная подвыборка
        if not (0.0 < self._data_fraction <= 1.0):
            raise ValueError(f"data_fraction должен быть в (0.0, 1.0], получено {self._data_fraction}")
        if self._data_fraction < 1.0:
            N = len(pdf_names)
            k = max(1, int(N * self._data_fraction))
            if self._data_seed is not None:
                rng = random.Random(self._data_seed)
                subset = rng.sample(pdf_names, k)
            else:
                subset = random.sample(pdf_names, k)
            if self.loger:
                self.loger(f"data_fraction={self._data_fraction}: {k}/{N} PDF")
            pdf_names = sorted(subset)

        self.count = len(pdf_names)
        self.pdf_names = pdf_names
        self.is_train = False

    def train(self):
        self.is_train = True
        
    def test(self):
        self.is_train = False

    def pdf2json_for_model(self, name_file):
        try:
            rez  = self.pred(self.pdf_dir/name_file)
            torch_dict = rez['torch_dict']
            
            if self.is_train:
                true_regions, true_category = self.coco_manager(name_file, rez['pdf_json'])   
                true_edges, true_nodes = self._get_true_edges(torch_dict,  rez['pdf_json']['rows'], true_regions, true_category) 
                torch_dict['true_edges'] = true_edges
                torch_dict['true_nodes'] = true_nodes
            
            del torch_dict['sp_A']
        except Exception as e:
            with open('failed_pdfs.log', 'a') as log:
                log.write(f"DATASET\t{name_file}\t{e}\n")
            print(f"ERROR in {name_file}: {e}")
            return {}
        return torch_dict


    def _get_true_edges(self, token, rows, region_segs, region_categories):
        def is_one_region(num_reg1, num_reg2):
            if num_reg1 == None:
                return 0
            if num_reg2 == None:
                return 0
            if num_reg1 == num_reg2:
                return 1
            return 0

        def get_category(seg, region_segs, region_categories):
            for r, c in zip(region_segs, region_categories):
                if seg.is_intersection(r):
                    return c
            return None

        def get_mini_seg(r):
            img_seg = ImageSegment(dict_p_size=r)
            if img_seg.height < 5:
                return img_seg
            delta = int(img_seg.height / 5)
            img_seg.y_bottom_right = img_seg.y_bottom_right - delta
            img_seg.y_top_left = img_seg.y_top_left + delta
            return img_seg

        row_segments = [get_mini_seg(row['segment']) for row in rows]
        A = token['inds']
        
        nums_regions = get_num_regions_of_rows(region_segs, row_segments)
        true_edges = [is_one_region(nums_regions[i], nums_regions[j]) for i, j in zip(A[0], A[1])]
        true_nodes = [get_category(row_seg, region_segs, region_categories) for row_seg in row_segments]
        return true_edges, true_nodes
    
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
                if self.loger:
                    self.loger("KEY ERROR FILES:")
                for i in key_error:
                    print(files[i])
                    if self.loger:
                        self.loger(files[i])

            if len(json_error) != 0:
                print("JSON ERROR FILES:")
                if self.loger:
                    self.loger("JSON ERROR FILES:")
                for i in json_error:
                    print(files[i])
                    if self.loger:
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
        return torch.tensor([vec_class(c) for c in classes], dtype=torch.float32, device=self.device)

    def _ensure_ram_cache(self):
        """Однократная eager-загрузка всех JSON в self._ram_cache."""
        if self._ram_cache is not None:
            return
        self._ram_cache = {}
        N = len(self.pdf_names)
        if self.loger:
            self.loger(f"ram_cache: загрузка {N} JSON в память...")
        for i, name in enumerate(self.pdf_names):
            try:
                self._ram_cache[name] = self.cacher(name)
            except Exception as e:
                print(f"ram_cache load error {name}: {e}")
                self._ram_cache[name] = {}
            if self.loger and (i + 1) % 5000 == 0:
                self.loger(f"ram_cache: {i+1}/{N} ({100*(i+1)/N:.1f}%)")
        if self.loger:
            size_mb = sum(len(json.dumps(v)) for v in self._ram_cache.values()) / 1024 / 1024
            self.loger(f"ram_cache: загружено {N} JSON ({size_mb:.1f} MB)")

    def __getitem__(self, idx):
        name_file = self.pdf_names[idx]
        
        # print(name_file)
        if not self.is_train:
            try :
                bboxes_true, classes_true = self._get_true_regions(idx)
                return {"bboxes_true":bboxes_true, 
                        "classes_true":classes_true,
                        "path": Path(self.pdf_dir, name_file)}
            except Exception as e:
                print(e)
                return {
                    "bboxes_true":[], 
                    "classes_true":[],
                    "path": Path(self.pdf_dir, name_file)
                }
        try:
            if self._ram_cache_flag:
                self._ensure_ram_cache()
                data = self._ram_cache.get(name_file, {})
            else:
                data = self.cacher(name_file)
        except Exception as e:
            print(e)
            print("cache error", name_file)
            return {}
        if len(data.keys()) == 0:
            return {}
        try:    
            data['X'] = self.to_tensor_safe(data['X'], torch.float32, self.device)
            data['Y'] = self.to_tensor_safe(data['Y'], torch.float32, self.device)
            N = data["N"]
            i = data['inds']
            index_for_mtrx = [i[0]+i[1], i[1]+i[0]]
            sp_A = torch.sparse_coo_tensor(indices=index_for_mtrx, values=[1 for e in index_for_mtrx[0]], size=(N, N), dtype=torch.float32, device=self.device)
            data['sp_A'] = sp_A
            data['true_edges'] = torch.tensor([0 if i is None else i for i in data['true_edges']], dtype=torch.float32, device=self.device)
            data['true_nodes'] = self.__class_to_vec(data['true_nodes'])
            data['file_name'] = '.'.join(name_file.split('.')[:-1])
            
        except Exception as e:
            print('data_error', name_file, e)
            return {}
        return data

    def to_tensor_safe(self, obj, dtype, device):
        if torch.is_tensor(obj):
            return obj.detach().clone().to(dtype=dtype, device=device)
        else:
            return torch.tensor(obj, dtype=dtype, device=device)

    def _get_true_regions(self, idx):
        
        name_file = self.pdf_names[idx]
        rez  = self.pred(self.pdf_dir/name_file)
        true_regions, classes_true = self.coco_manager(name_file, rez['pdf_json'])
        clean_bboxes, classes_true = self._clean_true_regions(true_regions, classes_true)
        bboxes_true =[reg.get_segment_p_size() for reg in clean_bboxes]
        return bboxes_true, classes_true

    def _clean_true_regions(self, true_regions, true_classes):
        clean_bboxes = []
        classes_true = []
        for cl, reg in zip(true_classes, true_regions):
            if reg.height > 3 and reg.width > 3:
                clean_bboxes.append(reg)
                classes_true.append(cl)
        return clean_bboxes, classes_true

    def init(self):
        N = self.count
        print("(init) SIZE DATASET: ", N)
        for i, d in enumerate(self):
            print(f"{(i+1)/N*100:4.2f} %", end='\r')

    
    
    def __str__(self):
        first_keys = self[0].keys()
        last_keys = self[-1].keys()

        def _safe_shape(d, key):
            return np.shape(d[key]) if key in d else "N/A"

        return f"""
            DATASET INFO:
            count row: {len(self)}
            cache_only: {self._cache_only}
            data_fraction: {self._data_fraction}
            ram_cache: {self._ram_cache_flag} (loaded={self._ram_cache is not None})
            first: {first_keys}
            \t A:{_safe_shape(self[0], 'sp_A')}
            \t nodes_feature:{_safe_shape(self[0], 'X')}
            \t edges_feature:{_safe_shape(self[0], 'Y')}
            \t true_edges:{_safe_shape(self[0], 'true_edges')}
            end:{last_keys}
            \t A:{_safe_shape(self[-1], 'sp_A')}
            \t nodes_feature:{_safe_shape(self[-1], 'X')}
            \t edges_feature:{_safe_shape(self[-1], 'Y')}
            \t true_edges:{_safe_shape(self[-1], 'true_edges')}
        """