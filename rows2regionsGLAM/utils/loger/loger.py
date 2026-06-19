import datetime
import os

class Loger:
    def __init__(self, LOG_FILE=None, log_dir=None):
        if LOG_FILE is None:
            LOG_FILE = f'log_{datetime.datetime.now().date()}.txt'
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            LOG_FILE = os.path.join(log_dir, LOG_FILE)
        self.log_file = LOG_FILE

    def __call__(self, str_):
        self.__log(str_)

    def __log(self, str_):
        with open(self.log_file, 'a') as f:
            f.write(str_+'\n')


    def time_log(self):
        str_ = "T: " + datetime.datetime.now().__str__()
        self.__log(str_)