import os
import sys
import warnings
from dotenv import load_dotenv

_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _path not in sys.path:
    sys.path.insert(0, _path)


def init():
    env_file = os.path.join(_path, '.env')
    load_dotenv(env_file)
    warnings.filterwarnings('ignore', message='Converting sparse tensor to CSR format')
    warnings.filterwarnings('ignore', message='Implicit dimension choice for softmax')
    warnings.filterwarnings('ignore', message='To copy construct from a tensor')
