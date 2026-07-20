#!/usr/bin/env python3
"""
Создание мини-датасетов из PubLayNet.

Режимы:
1) Одиночный: --n N → один датасет из N случайных PDF
2) Split: --split с --test-n и --val-n из одного COCO, непересекающиеся

Примеры:
  python scripts/create_mini_dataset.py --coco train.json --pdf-dir pdfs/train \\
      --output ablation/train --n 10000 --seed 42

  python scripts/create_mini_dataset.py --coco val.json --pdf-dir pdfs/dev \\
      --output ablation --split --test-n 1000 --val-n 1000 --seed 42
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path


def _write_subset(images, coco_meta, pdf_dir, output_dir, prefix, coco_stem):
    """Записать подмножество: pdf_out/{prefix}/ + coco_out/{prefix}_*.json."""
    pdf_out = output_dir / f'pdfs_{prefix}'
    pdf_out.mkdir(parents=True, exist_ok=True)

    selected_ids = {img['id'] for img in images}
    mini_annotations = []
    for anno_id, ann in enumerate(coco_meta.get('annotations', [])):
        if ann['image_id'] in selected_ids:
            ann_copy = ann.copy()
            ann_copy['id'] = anno_id
            mini_annotations.append(ann_copy)

    for img in images:
        img_name = img['file_name']
        if img_name.endswith('.jpg'):
            pdf_name = img_name[:-3] + 'pdf'
        elif img_name.endswith('.png'):
            pdf_name = img_name[:-3] + 'pdf'
        else:
            pdf_name = img_name
        src = Path(pdf_dir) / pdf_name
        dst = pdf_out / pdf_name
        shutil.copy2(src, dst)

    mini_coco = {k: v for k, v in coco_meta.items() if k not in ('images', 'annotations')}
    mini_coco['images'] = list(images)
    mini_coco['annotations'] = mini_annotations

    coco_out = output_dir / f'{coco_stem}_{prefix}.json'
    with open(coco_out, 'w') as f:
        json.dump(mini_coco, f)

    print(f'  {prefix}: {len(images)} PDFs → {pdf_out}')
    print(f'  COCO: {coco_out} ({len(mini_annotations)} annotations)')


def create_mini_dataset(coco_path, pdf_dir, output_dir, n_files=10, seed=42, prefix='train'):
    random.seed(seed)
    output_dir = Path(output_dir)
    pdf_out = output_dir / f'pdfs_{prefix}'
    pdf_out.mkdir(parents=True, exist_ok=True)

    with open(coco_path) as f:
        coco = json.load(f)

    images = coco.get('images', [])
    if len(images) < n_files:
        raise ValueError(f'COCO has {len(images)} images, need {n_files}')

    selected = random.sample(images, n_files)
    coco_stem = Path(coco_path).stem
    _write_subset(selected, coco, pdf_dir, output_dir, prefix, coco_stem)


def create_split_dataset(coco_path, pdf_dir, output_dir, test_n, val_n, seed=42):
    random.seed(seed)
    output_dir = Path(output_dir)

    with open(coco_path) as f:
        coco = json.load(f)

    images = coco.get('images', [])
    total = test_n + val_n
    if len(images) < total:
        raise ValueError(f'COCO has {len(images)} images, need {total} (test={test_n} + val={val_n})')

    selected = random.sample(images, total)
    test_images = selected[:test_n]
    val_images = selected[test_n:]

    coco_stem = Path(coco_path).stem
    # Мета-данные (categories, etc.) без images/annotations
    coco_meta = {k: v for k, v in coco.items() if k not in ('images', 'annotations')}
    # Аннотации копируем как есть — _write_subset их отфильтрует
    coco_meta['annotations'] = coco.get('annotations', [])

    _write_subset(test_images, coco_meta, pdf_dir, output_dir, 'test', coco_stem)
    _write_subset(val_images, coco_meta, pdf_dir, output_dir, 'val', coco_stem)

    print(f'Total: {total} PDFs (test={test_n}, val={val_n}), seed={seed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create mini/ablation datasets from PubLayNet')
    parser.add_argument('--coco', required=True, help='Path to COCO JSON')
    parser.add_argument('--pdf-dir', required=True, help='Path to PDF directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--n', type=int, default=10, help='Number of files (single mode)')
    parser.add_argument('--prefix', default='train', help='Output subfolder prefix (single mode: pdfs_{prefix}/)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', action='store_true',
                        help='Split mode: create disjoint test/val from same COCO')
    parser.add_argument('--test-n', type=int, default=1000, help='Test files (split mode)')
    parser.add_argument('--val-n', type=int, default=1000, help='Val files (split mode)')
    args = parser.parse_args()

    if args.split:
        create_split_dataset(args.coco, args.pdf_dir, args.output,
                             args.test_n, args.val_n, args.seed)
    else:
        create_mini_dataset(args.coco, args.pdf_dir, args.output,
                            args.n, args.seed, args.prefix)

