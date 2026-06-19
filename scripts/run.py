#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_BIN = os.environ.get("PYTHON_BIN", "/Users/macbookair/program/python/PageR/env/bin/python")

from dotenv import dotenv_values
from scripts.configs import resolve_env_file, write_dotenv, ENV_TEMPLATES


def run_experiment(experiment, env_path):
    env_vars = dict(dotenv_values(env_path))
    write_dotenv(env_path)

    start_script = PROJECT_ROOT / 'experiments' / experiment / 'start.py'
    if not start_script.exists():
        print(f'Experiment not found: {start_script}')
        sys.exit(1)

    print(f'=== Running: {experiment} ===')
    print(f'  Env:     {env_path}')
    print(f'  Device:  {env_vars.get("DEVICE", "?")}')
    print(f'  Train:   {env_vars.get("DATASET_PATH", "?")}')
    print(f'  Test:    {env_vars.get("TEST_PATH", "?")}')
    print(f'  Epochs:  {env_vars.get("EPOCHS", "?")}')
    print(f'  Script:  {start_script}')

    merged_env = os.environ.copy()
    merged_env.update(env_vars)
    pythonpath = str(PROJECT_ROOT)
    if "PYTHONPATH" in merged_env:
        pythonpath = f"{pythonpath}:{merged_env['PYTHONPATH']}"
    merged_env["PYTHONPATH"] = pythonpath

    subprocess.run([PYTHON_BIN, str(start_script)], env=merged_env, cwd=str(PROJECT_ROOT))

    src_csv = PROJECT_ROOT / 'result' / 'results.csv'
    if src_csv.exists():
        dst_dir = PROJECT_ROOT / 'results' / experiment
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(src_csv, dst_dir / 'results.csv')
        print(f'Moved results to results/{experiment}/results.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run rows2regionsGLAM experiments')
    parser.add_argument('experiment', help='Experiment name (folder under experiments/)')
    parser.add_argument(
        '--env', default='debug',
        help=f'Shortcut ({"/".join(ENV_TEMPLATES)}) or path to .env file'
    )
    args = parser.parse_args()

    env_file = resolve_env_file(args.env)
    run_experiment(args.experiment, env_file)
