import datetime

class Loger:
    def __init__(self, LOG_FILE):
        self.log_file = LOG_FILE

    def __call__(self, str_):
        self.__log(str_)

    def __log(self, str_):
        with open(self.log_file, 'a') as f:
            f.write(str_+'\n')


    def time_log(self):
        str_ = "T: " + datetime.datetime.now().__str__()
        self.__log(str_)