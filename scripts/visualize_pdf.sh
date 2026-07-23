#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <PDF_PATH>"
    exit 1
fi

PDF_PATH="$1"

if [ ! -f "$PDF_PATH" ]; then
    echo "Error: PDF file not found: $PDF_PATH"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

python scripts/visualize_document.py \
    --config config_glam.json \
    --model result/row2region_GLAM_no_font_seed_0_best \
    --pdf "$PDF_PATH" \
    --output output_.png \
    --tokenizer no_font \
    --dataset publaynet \
    --coco /home/dataset/ablation_10k/val.json \
    --dpi 300 \
    --page 0