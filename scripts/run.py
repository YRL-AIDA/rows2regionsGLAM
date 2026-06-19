#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.configs import ALL_CONFIGS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write_env(mode):
    config = ALL_CONFIGS[mode]
    env_path = PROJECT_ROOT / '.env'
    with open(env_path, 'w') as f:
        for k, v in config.items():
            f.write(f"{k}='{v}'\n")
    print(f'Wrote .env for mode: {mode}')


def run_experiment(experiment, mode):
    if mode not in ALL_CONFIGS:
        print(f'Unknown mode: {mode}. Available: {list(ALL_CONFIGS.keys())}')
        sys.exit(1)

    config = ALL_CONFIGS[mode]
    env_vars = os.environ.copy()
    for k, v in config.items():
        env_vars[k] = str(v)

    write_env(mode)

    start_script = PROJECT_ROOT / 'experiments' / experiment / 'start.py'
    if not start_script.exists():
        print(f'Experiment not found: {start_script}')
        sys.exit(1)

    print(f'=== Running: {experiment} (mode: {mode}) ===')
    print(f'  Device:  {config["DEVICE"]}')
    print(f'  Train:   {config["DATASET_PATH"]}')
    print(f'  Test:    {config["TEST_PATH"]}')
    print(f'  Epochs:  {config.get("EPOCHS", "default")}')
    print(f'  Script:  {start_script}')

    subprocess.run([sys.executable, str(start_script)], env=env_vars, cwd=str(PROJECT_ROOT))

    src_csv = PROJECT_ROOT / 'experiments' / experiment / 'result' / 'results.csv'
    if src_csv.exists():
        dst_dir = PROJECT_ROOT / 'results' / experiment
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_csv, dst_dir / 'results.csv')
        print(f'Copied results to results/{experiment}/results.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run rows2regionsGLAM experiments')
    parser.add_argument('experiment', help='Experiment name (folder under experiments/)')
    parser.add_argument('--mode', choices=list(ALL_CONFIGS.keys()),
                        default='debug', help='Run mode')
    args = parser.parse_args()
    run_experiment(args.experiment, args.mode)
