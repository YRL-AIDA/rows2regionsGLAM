#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
from pathlib import Path

def create_mini_dataset(coco_path, pdf_dir, output_dir, n_files=10, seed=42):
    random.seed(seed)
    output_dir = Path(output_dir)
    pdf_out = output_dir / 'pdfs'
    pdf_out.mkdir(parents=True, exist_ok=True)

    with open(coco_path) as f:
        coco = json.load(f)

    images = coco.get('images', [])
    if len(images) < n_files:
        raise ValueError(f'COCO has {len(images)} images, need {n_files}')

    selected = random.sample(images, n_files)

    mini_images = []
    mini_annotations = []
    anno_id = 0
    selected_ids = {img['id'] for img in selected}

    for img in selected:
        jpg_name = img['file_name']
        pdf_name = jpg_name[:-3] + 'pdf' if jpg_name.endswith('.jpg') else jpg_name
        src = Path(pdf_dir) / pdf_name
        dst = pdf_out / pdf_name
        shutil.copy2(src, dst)
        mini_images.append(img)

    for ann in coco.get('annotations', []):
        if ann['image_id'] in selected_ids:
            ann_copy = ann.copy()
            ann_copy['id'] = anno_id
            anno_id += 1
            mini_annotations.append(ann_copy)

    mini_coco = {k: v for k, v in coco.items() if k not in ('images', 'annotations')}
    mini_coco['images'] = mini_images
    mini_coco['annotations'] = mini_annotations

    coco_out = output_dir / f'{Path(coco_path).stem}_mini.json'
    with open(coco_out, 'w') as f:
        json.dump(mini_coco, f)

    print(f'Created mini dataset: {len(mini_images)} PDFs in {pdf_out}')
    print(f'Mini COCO: {coco_out} ({len(mini_annotations)} annotations)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a mini debug dataset from PublayNet')
    parser.add_argument('--coco', required=True, help='Path to COCO JSON (train.json or val.json)')
    parser.add_argument('--pdf-dir', required=True, help='Path to PDF directory')
    parser.add_argument('--output', required=True, help='Output directory for mini dataset')
    parser.add_argument('--n', type=int, default=10, help='Number of files')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    create_mini_dataset(args.coco, args.pdf_dir, args.output, args.n, args.seed)
