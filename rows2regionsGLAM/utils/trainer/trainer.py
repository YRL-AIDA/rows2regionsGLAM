import time
import os
import numpy as np
import torch
import torch.nn as nn
from ...models import get_loss, get_model
from dotenv import load_dotenv

# env_file = os.path.join('..', '.env')
# load_dotenv(env_file)

class Trainer:
    def __init__(self, **conf):
        if "loger" not in conf.keys():
            raise Exception('Создайте и передайте логер "loger": Loger(path))')
        else:
            self.loger = conf['loger']
            
        if "train_param" not in conf.keys():
            raise Exception('Создайте и передайте параметры "train_param"')
        else:
            self.train_param = conf['train_param']
            
        if "model" not in conf.keys():
            raise Exception('Создайте и передайте модель "model"')
        else:
            self.model = conf['model']
            
        if "dataset" not in conf.keys():
            raise Exception('Создайте и передайте датасет "dataset"')
        else:
            self.dataset = conf['dataset']

        if "loss" not in conf.keys():
            raise Exception('Создайте и передайте loss function "loss"')
        else:
            self.loss = conf['loss']
            
        self.loger.time_log()
        self.loger("Create Trainer")
        self.device = torch.device(os.environ.get('DEVICE', 'cpu'))

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
                if len(data_graph_dict['X']) < 2:
                    continue
                if len(data_graph_dict['Y']) == 0:
                    continue
                
                pred_graph_dict = model(data_graph_dict)
                loss = criterion(pred_graph_dict, data_graph_dict)
                my_loss_list.append(loss.item())
                print(f"{(j+1)/len(batch)*100:.2f} % Batch loss={my_loss_list[-1]:.4f}" + " "*40, end="\r")
            except Exception as e:
                print(e)
                if "Y" in data_graph_dict.keys():
                    print(np.array(data_graph_dict['Y'].cpu()).shape)
                if "X" in data_graph_dict.keys():
                    print(np.array(data_graph_dict['X'].cpu()).shape)
                continue
            if train:  
                loss.backward()
        if train:
            optimizer.step()
        return np.mean(my_loss_list)

    def _train_model(self, save_frequency=5, start_epoch=0):  
        model=self.model
        dataset=self.dataset
        criterion=self.loss
        batch_size = self.train_param["batch_size"]
        count_epochs = self.train_param["epochs"]
        save_frequency = self.train_param['save_frequency']
        restart_num = self.train_param['restart_num']
        
        start_epoch = 0 if restart_num is None else (restart_num+1)*save_frequency
        
        optimizer = torch.optim.Adam(
        list(model.parameters()),
            lr=self.train_param["learning_rate"],
        )
        model.to(self.device)
        criterion.to(self.device)

        loss_list = []
        start = time.time()
        train_dataset, val_dataset = self._split_index_train_val(dataset, val_split=0.1, batch_size=batch_size)
        for k in range(start_epoch, count_epochs):
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
                torch.save(model.state_dict(), self.model.path+f"_tmp_{num}")
        self.loger(f"Время обучения: {time.time()-start:.2f} сек")
        

    def start_train(self):
        self.dataset.train()
        self.loger.time_log()
        str_ = self.dataset.__str__()
        str_ += '\n'.join(f"{key}:\t{val}" for key, val in self.train_param.items())
        print(str_)
        self.loger(str_)
       
        
        self._train_model()