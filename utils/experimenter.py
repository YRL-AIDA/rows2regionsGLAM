from typing  import Dict
from pathlib import Path
import pandas as pd


class Experimenter:
    def __init__(self, name, result_save_path):
        self.result_sava_path = Path(result_save_path)
        self.name = name
        if not self.result_sava_path.exists():
            self.result_sava_path.mkdir()

    def experiment(self, 
                   fun_get_model_with_param, 
                   fun_get_dataset_with_param,
                   fun_train_model_with_param,
                   fun_test_model_with_param,
                   fun_result_to_row,
                   dict_params: Dict,
                   fun_save_files_exp=None):
        """
        fun_get_model_with_param: функция, которая возвращает модель с заданными параметрами
        fun_get_dataset_with_param: функция, которая возвращает датасет с заданными параметрами
        fun_train_model_with_param: функция, которая обучает модель на датасете, возращает нужные данные 
        fun_test_model_with_param: функция, которая тестирует модель на датасете, возращает нужные данные 
        fun_result_to_row: функция, которая преобразует данные в строку для записи в файл

        dict_params: словарь с параметрами для модели и датасета 
        {
            "name_exp": {
                "model_param": {}
                "dataset_param": {}
                "train_param": {}
                "test_param": {}
            }
        }


        fun_save_files_exp: функция, которая сохраняет файлы эксперимента
        """
        results = []
        for name_exp, param in dict_params.items():
            model = fun_get_model_with_param(param=param["model_param"])
            dataset = fun_get_dataset_with_param(param=param["dataset_param"])
            train_result = fun_train_model_with_param(model=model, dataset=dataset, param=param["train_param"])
            test_result = fun_test_model_with_param(model=model, dataset=dataset, param=param["test_param"])
            res_exp = fun_result_to_row(train_result, test_result)
            res_exp['name'] = name_exp
            results.append(res_exp)
            if fun_save_files_exp:
                fun_save_files_exp(base_path=self.result_sava_path/'output'/name_exp,
                                   model=model, 
                                   dataset=dataset, 
                                   train_result=train_result, 
                                   test_result=test_result)
        self.save_results( results)

        
    def save_results(self, results):
        df = pd.DataFrame(results)
        df.to_csv(self.result_sava_path/'results.csv')



        

