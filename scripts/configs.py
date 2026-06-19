import os
from pathlib import Path
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_TEMPLATES = {
    'debug': '.env.debug',
    'local': '.env.local',
    'server': '.env.server',
}


def resolve_env_file(env_arg):
    if env_arg in ENV_TEMPLATES:
        path = PROJECT_ROOT / ENV_TEMPLATES[env_arg]
    else:
        path = Path(env_arg)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f'Env file not found: {path}')
    return path


def write_dotenv(env_path):
    dest = PROJECT_ROOT / '.env'
    with open(env_path) as src, open(dest, 'w') as dst:
        dst.write(src.read())
    return dest
