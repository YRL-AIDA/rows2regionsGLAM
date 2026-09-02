#!/usr/bin/env bash
#
# batch_visual_data.sh
# --------------------
# Обходит все PDF в DIR_PDF, для каждого вызывает
#   python scripts/visual_data_for_model.py <имя.pdf> --env local
# и сохраняет HTML-отчёты в OUTPUT_DIR.
#
# Использование:
#   ./batch_visual_data.sh
#   DIR_PDF=/path/to/pdfs OUTPUT_DIR=./reports ./batch_visual_data.sh
# ------------------------------------------------------------------

set -euo pipefail

# ── Настройки (можно переопределить через переменные окружения) ──────
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPTS_DIR")"

DIR_PDF="${DIR_PDF:-/home/dataset/ablation_10k/pdfs_val}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/visual_reports}"
ENV="${ENV:-local}"

mkdir -p "$OUTPUT_DIR"

# Статистика
total=0
success=0
failed=0
log_file="$OUTPUT_DIR/_batch_errors.log"
: > "$log_file"

echo "============================================"
echo "PDF dir:    $DIR_PDF"
echo "Output dir: $OUTPUT_DIR"
echo "Env:        $ENV"
echo "============================================"

# ── Основной цикл ────────────────────────────────────────────────────
shopt -s nullglob
for pdf_path in "$DIR_PDF"/*.pdf; do
    pdf_name="$(basename "$pdf_path")"
    stem="${pdf_name%.pdf}"
    output_html="$OUTPUT_DIR/${stem}.html"
    total=$((total + 1))

    echo "[${total}] $pdf_name ... "

    if [ -f "$output_html" ]; then
        echo "  → already exists, skipping"
        success=$((success + 1))
        continue
    fi

    if python "$SCRIPTS_DIR/visual_data_for_model.py" \
        "$pdf_name"                  \
        --env "$ENV"                \
        --dir_pdf "$DIR_PDF"        \
        --output "$output_html"     \
        2>&1 | tail -1; then

        success=$((success + 1))
    else
        failed=$((failed + 1))
        echo "  ✗ ERROR (see $log_file)"
        echo "$pdf_name" >> "$log_file"
    fi
done

# ── Итог ─────────────────────────────────────────────────────────────
echo "============================================"
echo "Done.  Total: $total  Success: $success  Failed: $failed"
echo "Reports: $OUTPUT_DIR"
if [ "$failed" -gt 0 ]; then
    echo "Errors:  $log_file"
fi