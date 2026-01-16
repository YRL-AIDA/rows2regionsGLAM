from abc import ABC, abstractmethod
from typing import Dict, List

class BaseTokenizer(ABC):
    @abstractmethod
    def get_dict_vec(self) -> Dict[str, List[int]]:
        """
        Получение словаря признаков
        """
        pass
    @abstractmethod
    def get_name(self) -> str:
        pass 