import time
import os
import numpy as np
import torch
import torch.nn as nn
from ...models.rowGLAM_base import CustomLossBase, TorchModelBase
from ...models.rowGLAM_custom import CustomLoss, TorchModel
from dotenv import load_dotenv
env_file = os.path.join('..', '.env')
load_dotenv(env_file)

class Trainer:
    def __init__(self, conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
        if "params" not in conf.keys():
            raise Exception('Создайте и передайте параметры "params"')
        else:
            self.params = conf['params']
        if "model_name" not in conf.keys():
            raise Exception('Создайте и передайте название модели "model_name"')
        else:
            self.model_name = conf['model_name']
        self.loger.time_log()
        self.loger("Create Trainer")
        self.device = torch.device(os.environ.get('DEVICE', 'cpu'))
        self.rowGlam_type = os.environ.get('ROW_GLAM_TYPE', 'base')

    def _validation(self, model, batch, criterion):     
        return self._step(model, batch, optimizer=None, criterion=criterion, train=False)

    def _split_index_train_val(self, dataset, val_split=0.2, shuffle=True, seed=1234,batch_size=64):
        N = len(dataset)
        count_batchs = int(N*(1-val_split))//batch_size
        count_val_batch = int(N*(val_split))//batch_size
        train_size = count_batchs * batch_size 
        indexs = [i for i in range(N)]
        if shuffle:
            np.random.shuffle(indexs)
        train_indexs = indexs[:train_size]
        val_indexs = indexs[train_size:]
        batchs_train_indexs = [[train_indexs[k*batch_size+i] for i in range(batch_size)] for k in range(count_batchs)]
        batchs_val_indexs = [[val_indexs[k*batch_size+i] for i in range(batch_size)] for k in range(count_val_batch)]
        return batchs_train_indexs, batchs_val_indexs    

    def _step(self, model: torch.nn.Module, batch, optimizer, criterion, train=True):
        if train:
            optimizer.zero_grad()
        my_loss_list = []
    
        for j, data_graph_dict in enumerate(batch):
            try:
                if data_graph_dict is None:
                    continue
                pred_graph_dict = model(data_graph_dict)
                loss = criterion(pred_graph_dict, data_graph_dict)
                my_loss_list.append(loss.item())
                print(f"{(j+1)/len(batch)*100:.2f} % Batch loss={my_loss_list[-1]:.4f}" + " "*40, end="\r")
            except Exception as e:
                print(e)
                if "Y" in data_graph_dict.keys():
                    print(np.array(data_graph_dict['Y']).shape)
                if "X" in data_graph_dict.keys():
                    print(np.array(data_graph_dict['X']).shape)
                continue
            if train:  
                loss.backward()
        if train:
            optimizer.step()
        return np.mean(my_loss_list)

    def _train_model(self, model, dataset, save_frequency=5, start_epoch=0):  
        optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=self.params["learning_rate"],
        )
        if self.rowGlam_type == "base":
            criterion = CustomLossBase(self.params["loss_params"])
        else:
            criterion = CustomLoss(self.params["loss_params"])

        model.to(self.device)
        criterion.to(self.device)

        loss_list = []
        start = time.time()
        train_dataset, val_dataset = self._split_index_train_val(dataset, val_split=0.1, batch_size=self.params["batch_size"])
        for k in range(start_epoch, self.params["epochs"]):
            my_loss_list = []
            if k == start_epoch:
                start = time.time()
            for l, batch_indexs in enumerate(train_dataset):
                batch = [dataset[ind] for ind in batch_indexs]
                batch_loss = self._step(model, batch, optimizer, criterion)
                my_loss_list.append(batch_loss)
                print(f"Batch # {l+1} loss={my_loss_list[-1]:.4f}" + " "*40, end='\r')
                if (k == start_epoch and l==0):
                    print(f"Время обучения batch'а {time.time()-start:.2f} сек")
            train_val = np.mean(my_loss_list)
            loss_list.append(train_val)

            my_loss_list = []
            for l, batch_indexs in enumerate(val_dataset):
                batch = [dataset[ind] for ind in batch_indexs]
                batch_loss = self._validation(model, batch, criterion)
                my_loss_list.append(batch_loss)
                print(f"Batch # {l+1} loss={my_loss_list[-1]:.4f}" + " "*40, end='\r')
            validation_val =  np.mean(my_loss_list)
            print("="*10, f"EPOCH #{k+1}","="*10, f"({train_val:.4f}/{validation_val:.4f})")
            if k == start_epoch:
                print(f"Время обучения epoch {time.time()-start:.2f} сек")    
                
            self.loger(f"EPOCH #{k}\t {train_val:.8f} (VAL: {validation_val:.8f})")  
            if (k+1) % save_frequency == 0:
                num = k//save_frequency
                torch.save(model.state_dict(), self.model_name+f"_tmp_{num}")
        self.loger(f"Время обучения: {time.time()-start:.2f} сек")
        torch.save(model.state_dict(), self.model_name)


    def _load_checkpoint(self, model, path_model,restart_num=None):
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
        print(checkpoint_path)
        return restart_num

    def start_train(self, save_frequency, dataset, is_restart = False, restart_num = None):
        if is_restart:
            self.loger("R E S T A R T ")
        self.loger.time_log()
        try:
            str_ = dataset.__str__()
            str_ += '\n'.join(f"{key}:\t{val}" for key, val in self.params.items())
            print(str_)
            if not is_restart:
                self.loger(str_)
        except:
            print(dataset)
        self.params['sigmoidEdge'] = False
        if self.rowGlam_type == "base":
            model:torch.nn.Module = TorchModelBase(self.params)
        else:
            model: torch.nn.Module = TorchModel(self.params)
        if is_restart:
            restart_num = self._load_checkpoint(model, self.model_name)
        
        start_epoch = 0 if restart_num is None else (restart_num+1)*save_frequency
        self._train_model(model, dataset, save_frequency=save_frequency, start_epoch=start_epoch)