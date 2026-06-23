#!/usr/bin/env python3
"""
Построение кеша датасета через multiprocessing Pool.
При --verify сравнивает parser-стабильные поля (N, X, Y, inds)
с эталонным кешем debug_data — проверка, что pagerlib выдаёт
идентичный результат на сервере и локальной машине.

Использование:
    python scripts/build_cache.py train --env server
    python scripts/build_cache.py train --env server --verify debug_data/cache
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import dotenv_values

from rows2regionsGLAM.utils.loger import Loger
from rows2regionsGLAM.utils.coco_manager import COCOManager
from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.datasetloaders.base_line_dataset import GLAMDataset

PARSER_KEYS = ['N', 'X', 'Y', 'inds']


_worker_ds = None


def _init_worker(pdf_dir, coco_path, name_dataset, cache_dir, is_train):
    global _worker_ds
    coco_manager = COCOManager(loger=None, coco_path=coco_path, name_dataset=name_dataset)
    pred = PredProcessor(loger=None)
    _worker_ds = GLAMDataset(
        coco_manager=coco_manager,
        default_index=0,
        pred=pred,
        cache_dir=cache_dir,
        pdf_dir=pdf_dir,
    )
    if is_train:
        _worker_ds.train()


def _cache_one(idx):
    _worker_ds[idx]


def _resolve(value):
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


def load_debug_refs(verify_path):
    debug_dir = Path(verify_path)
    if not debug_dir.is_absolute():
        debug_dir = PROJECT_ROOT / debug_dir
    if not debug_dir.exists():
        return None, f"WARNING: verify path not found: {debug_dir}"
    refs = {}
    for f in debug_dir.glob('*.json'):
        with open(f) as fh:
            key = f.stem
            if key.endswith('.pdf'):
                key = key[:-4]
            refs[key] = {k: v for k, v in json.load(fh).items() if k in PARSER_KEYS}
    return refs, None


def compare(ref, live, tol=1e-5):
    diffs = []
    for key in PARSER_KEYS:
        a = ref.get(key)
        b = live.get(key)
        if a is None or b is None:
            diffs.append(f"  missing key '{key}'")
            continue
        if key == 'N':
            if a != b:
                diffs.append(f"  N: ref={a} live={b}")
        elif key in ('X', 'Y'):
            if len(a) != len(b):
                diffs.append(f"  len({key}): ref={len(a)} live={len(b)}")
            elif len(a) > 0 and len(a[0]) != len(b[0]):
                diffs.append(f"  dim({key}[0]): ref={len(a[0])} live={len(b[0])}")
            else:
                mismatch = sum(
                    1 for i in range(len(a)) for j in range(len(a[i]))
                    if abs(a[i][j] - b[i][j]) > tol
                )
                if mismatch:
                    diffs.append(f"  {key}: {mismatch} value diffs > {tol}")
        elif key == 'inds':
            if len(a) != 2 or len(b) != 2:
                diffs.append(f"  inds shape: ref={len(a)} live={len(b)}")
            elif len(a[0]) != len(b[0]):
                diffs.append(f"  len(inds[0]): ref={len(a[0])} live={len(b[0])}")
            elif a != b:
                diffs.append("  inds: graph structure differs")
    return diffs


def build(env_path, mode, verify_path):
    env_vars = dict(dotenv_values(env_path))

    if mode == 'train':
        pdf_dir = _resolve(env_vars['DATASET_PATH'])
        coco_path = _resolve(env_vars['COCO_PATH'])
        name = env_vars.get('NAME_DATASET', 'publaynet')
    else:
        pdf_dir = _resolve(env_vars['TEST_PATH'])
        coco_path = _resolve(env_vars['TEST_COCO_PATH'])
        name = env_vars.get('NAME_TEST_DATASET', 'publaynet')

    cache_dir = _resolve(env_vars['CASH_PDF_PATH'])

    loger = Loger()
    coco_manager = COCOManager(loger=loger, coco_path=coco_path, name_dataset=name)
    pred = PredProcessor(loger=loger)

    debug_refs, warn = None, None
    if verify_path:
        debug_refs, warn = load_debug_refs(verify_path)
        if warn:
            print(warn)
        else:
            print(f"Loaded {len(debug_refs)} debug reference(s) from {verify_path}")
            in_set = [n for n in debug_refs if f"{n}.pdf" in set(os.listdir(pdf_dir))]
            print(f"  Present in dataset: {len(in_set)}/{len(debug_refs)}")

    dataset = GLAMDataset(
        coco_manager=coco_manager,
        default_index=0,
        pred=pred,
        loger=loger,
        cache_dir=cache_dir,
        pdf_dir=pdf_dir,
    )

    print(f"Mode:   {mode}")
    print(f"PDFs:   {pdf_dir}")
    print(f"Cache:  {cache_dir}")
    print(f"Files:  {len(dataset)}")
    print()

    dataset.train()
    total = len(dataset)

    try:
        from tqdm import tqdm
        with Pool(
            cpu_count(),
            initializer=_init_worker,
            initargs=(pdf_dir, coco_path, name, cache_dir, mode == 'train'),
        ) as pool:
            for _ in tqdm(pool.imap_unordered(_cache_one, range(total)), total=total):
                pass
    except ImportError:
        with Pool(
            cpu_count(),
            initializer=_init_worker,
            initargs=(pdf_dir, coco_path, name, cache_dir, mode == 'train'),
        ) as pool:
            pool.map(_cache_one, range(total))

    print()

    if not debug_refs:
        print("Done. No verification requested (use --verify debug_data/cache).")
        return True

    errors = []
    ok = 0
    for i, pdf_name in enumerate(dataset.pdf_names):
        stem = Path(pdf_name).stem
        if stem not in debug_refs:
            continue
        cache_path = Path(cache_dir) / f"{pdf_name}.json"
        if not cache_path.exists():
            errors.append(f"{pdf_name}: cache file not found")
            continue
        with open(cache_path) as f:
            live = {k: v for k, v in json.load(f).items() if k in PARSER_KEYS}
        diffs = compare(debug_refs[stem], live)
        if diffs:
            errors.append(f"{pdf_name}:")
            errors.extend(diffs)
        else:
            ok += 1
            print(f"  ✓ {pdf_name}")

    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    print(f"Matched:  {ok}/{len([n for n in dataset.pdf_names if Path(n).stem in debug_refs])}")
    print(f"Errors:   {len([e for e in errors if not e.startswith('  ')])}")
    if errors:
        print("\nDetails:")
        for e in errors:
            print(e)

    return len(errors) == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataset cache via GLAMDataset.init()')
    parser.add_argument('mode', choices=['train', 'test'],
                        help='train (with COCO labels) or test (parser only)')
    parser.add_argument('--env', default='debug',
                        help='Env shortcut or path to .env')
    parser.add_argument('--verify', default=None,
                        help='Path to debug_data/cache for parser stability check')
    args = parser.parse_args()

    from scripts.configs import resolve_env_file

    env_path = resolve_env_file(args.env)
    ok = build(env_path, args.mode, args.verify)
    sys.exit(0 if ok else 1)
