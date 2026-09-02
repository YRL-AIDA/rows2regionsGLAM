#!/usr/bin/env python3
"""
Генерация val.json с «идеальными» регионами, построенными из полносвязного графа
на основе COCO-разметки.

Идея (из эксперимента p07_diff_graphs):
1. Парсим PDF → получаем строки (rows).
2. Строим полносвязный граф между всеми строками (AllRowGLAMTokenizer).
3. По пересечению строк с true-регионами из COCO определяем:
   - класс каждой строки (категория региона, которому она принадлежит);
   - true-рёбра: строки в одном регионе → ребро 1, в разных → ребро 0 (удалить).
4. Собираем связные компоненты графа (только рёбра со значением 1) → «идеальные» регионы.
5. Сохраняем их в COCO-формате (val.json).

Преимущество: регионы собираются из реальных границ строк, а не из bbox-ов COCO,
поэтому исчезают ошибки расхождения границ (пиксельных несовпадений).

Использование:
    python scripts/ideal_regions_to_coco.py \
        --pdf-dir /home/dataset/ablation_10k/pdfs_val \
        --coco /home/dataset/ablation_10k/val.json \
        --output /home/dataset/ablation_10k/val_ideal.json \
        --dataset-name publaynet \
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import signal
import sys
import time
from collections import Counter
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── project root ───────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ── shared globals for parallel workers ─────────────────────────────────
_worker_pdf_dir = None
_worker_coco_path = None
_worker_dataset_name = None
_worker_categories = None          # List[dict] — категории из COCO
_worker_id2name = None             # Dict[int, str] — id → имя класса
_worker_name2id = None             # Dict[str, int] — имя класса → id
_worker_coco_annotations = None    # Dict[str, List[dict]] — аннотации COCO по именам PDF

ORACLE_TIMEOUT = 300


def _timeout_handler(signum, frame):
    raise TimeoutError("oracle timeout")


def _init_worker(pdf_dir, coco_path, dataset_name):
    global _worker_pdf_dir, _worker_coco_path, _worker_dataset_name
    global _worker_categories, _worker_id2name, _worker_name2id
    global _worker_coco_annotations

    signal.signal(signal.SIGALRM, _timeout_handler)
    pid = os.getpid()
    time.sleep((pid % 10) * 0.3)

    _worker_pdf_dir = pdf_dir
    _worker_coco_path = coco_path
    _worker_dataset_name = dataset_name

    # ── загружаем COCO один раз в процессе ──
    with open(coco_path, "r") as f:
        coco = json.load(f)

    _worker_categories = coco.get("categories", [])

    id2name: Dict[int, str] = {}
    for cat in _worker_categories:
        id2name[cat["id"]] = cat["name"]
    id2name[0] = "other"
    _worker_id2name = id2name
    _worker_name2id = {v: k for k, v in id2name.items()}

    # ── мапинг image_id → имя PDF ──
    id_to_file: Dict[int, str] = {}
    for img in coco.get("images", []):
        fname = img["file_name"]
        if fname.endswith(".jpg") or fname.endswith(".png"):
            fname = fname[:-3] + "pdf"
        id_to_file[img["id"]] = fname

    # ── группируем аннотации по имени PDF ──
    pdf_ann: Dict[str, List[dict]] = {}
    for ann in coco.get("annotations", []):
        pdf_name = id_to_file.get(ann["image_id"])
        if pdf_name is None:
            continue
        pdf_ann.setdefault(pdf_name, []).append(ann)
    _worker_coco_annotations = pdf_ann


def _get_true_regions(pdf_name: str, page_info: Dict) -> Tuple[List, List]:
    """Возвращает списки ImageSegment и category_id для заданного PDF."""
    from pagerlib.dtypes import ImageSegment

    if _worker_dataset_name == "doclaynet":
        coef_w = page_info["width"] / 1025
        coef_h = page_info["height"] / 1025
    else:
        coef_w, coef_h = 1.0, 1.0

    annotations = _worker_coco_annotations.get(pdf_name, [])
    region_segs = []
    region_categories = []

    for ann in annotations:
        bbox = ann["bbox"]  # [x, y, w, h]
        seg_dict = {
            "x_top_left": int(bbox[0] * coef_w),
            "y_top_left": int(bbox[1] * coef_h),
            "width": int(bbox[2] * coef_w),
            "height": int(bbox[3] * coef_h),
        }
        if seg_dict["height"] <= 0 or seg_dict["width"] <= 0:
            continue
        region_segs.append(ImageSegment(dict_p_size=seg_dict))
        region_categories.append(ann["category_id"])

    return region_segs, region_categories


def _compute_true_edges_and_nodes(
    inds: Tuple[List[int], List[int]],
    rows_json: List[dict],
    region_segs: List,
    region_categories: List[int],
) -> Tuple[List[int], List[Optional[int]]]:
    """Вычисляет true-рёбра (1=соединить, 0=удалить) и классы узлов."""
    from pagerlib.dtypes import ImageSegment
    from rows2regionsGLAM.utils.intersect_util import get_num_regions_of_rows

    def get_mini_seg(r):
        img_seg = ImageSegment(dict_p_size=r)
        if img_seg.height < 5:
            return img_seg
        delta = int(img_seg.height / 5)
        img_seg.y_bottom_right -= delta
        img_seg.y_top_left += delta
        return img_seg

    def get_category(seg, region_segs, region_categories):
        for r, c in zip(region_segs, region_categories):
            if seg.is_intersection(r):
                return c
        return None

    def is_one_region(num_reg1, num_reg2):
        if num_reg1 is None or num_reg2 is None:
            return 0
        return 1 if num_reg1 == num_reg2 else 0

    row_segments = [get_mini_seg(row["segment"]) for row in rows_json]
    nums_regions = get_num_regions_of_rows(region_segs, row_segments)

    true_edges = [
        is_one_region(nums_regions[i], nums_regions[j])
        for i, j in zip(inds[0], inds[1])
    ]
    true_nodes = [
        get_category(row_seg, region_segs, region_categories)
        for row_seg in row_segments
    ]
    return true_edges, true_nodes


def _compute_ideal_regions(
    rows_json: List[dict],
    inds: Tuple[List[int], List[int]],
    true_edges: List[int],
    true_nodes: List[Optional[int]],
) -> List[dict]:
    """
    Строит идеальные регионы через связные компоненты графа
    (аналог Rows2Regions.regions_from_graph).

    Возвращает список словарей с ключами:
        rows_json  — список строк региона,
        bbox       — [x, y, w, h],
        category_id — числовой id категории.
    """
    from pagerlib.dtypes import ImageSegment, Region
    from pagerlib.dtypes.relationship import Graph

    graph = Graph()
    for row in rows_json:
        seg = ImageSegment(dict_p_size=row["segment"])
        graph.add_node(*seg.get_center())

    for ni, nj, is_edge in zip(inds[0], inds[1], true_edges):
        if is_edge:
            graph.add_edge(ni + 1, nj + 1)  # Graph нумерует узлы с 1

    regions: List[Region] = []
    for component in graph.get_related_graphs():
        indexes = [n.index - 1 for n in component.get_nodes()]
        labels = [true_nodes[i] or 0 for i in indexes]
        majority = Counter(labels).most_common(1)[0][0]
        regions.append(
            Region(
                children=[rows_json[i] for i in indexes],
                data={"label": _worker_id2name.get(majority, "other")},
            )
        )

    result = []
    for reg in regions:
        seg = reg.segment
        label = reg.data.get("label", "other")
        cat_id = _worker_name2id.get(label, 0)
        result.append({
            "rows_json": reg.children,
            "bbox": [
                seg.x_top_left,
                seg.y_top_left,
                seg.width,
                seg.height,
            ],
            "category_id": cat_id,
        })
    return result


def _process_one(pdf_name: str) -> Optional[dict]:
    """Обрабатывает один PDF и возвращает словарь-результат или None при ошибке."""
    signal.alarm(ORACLE_TIMEOUT)
    try:
        from experiments.paper.p07_diff_graphs.pipeline import AllRowGLAMTokenizer
        from rows2regionsGLAM.pred_processor import PredProcessor

        pdf_path = Path(_worker_pdf_dir) / pdf_name
        tokenizer = AllRowGLAMTokenizer()
        pred = PredProcessor(tokenizer=tokenizer)

        # 1. Парсим PDF, строим полносвязный граф
        rez = pred(pdf_path)
        torch_dict = rez["torch_dict"]
        pdf_json = rez["pdf_json"]
        rows_json = pdf_json["rows"]
        inds = torch_dict["inds"]

        # 2. Получаем true-регионы из COCO
        region_segs, region_categories = _get_true_regions(pdf_name, pdf_json)

        # 3. Вычисляем true-рёбра и классы узлов
        true_edges, true_nodes = _compute_true_edges_and_nodes(
            inds, rows_json, region_segs, region_categories
        )

        # 4. Строим идеальные регионы
        ideal_regions = _compute_ideal_regions(
            rows_json, inds, true_edges, true_nodes
        )

        return {
            "pdf_name": pdf_name,
            "width": pdf_json["width"],
            "height": pdf_json["height"],
            "regions": ideal_regions,
        }

    except TimeoutError:
        print(f"SKIP (timeout >{ORACLE_TIMEOUT}s): {pdf_name}")
        return None
    except KeyError:
        # PDF нет в COCO — это нормально для некоторых датасетов
        return None
    except Exception as e:
        print(f"SKIP (error): {pdf_name} — {e}")
        return None
    finally:
        signal.alarm(0)


def _collect_pdf_names(pdf_dir: str) -> List[str]:
    """Собирает отсортированный список имён PDF в директории."""
    pdf_dir = Path(pdf_dir)
    names = sorted(
        f for f in os.listdir(pdf_dir)
        if f.endswith(".pdf") and not os.path.isdir(pdf_dir / f)
    )
    return names


def _build_coco(
    results: List[dict],
    categories: List[dict],
) -> dict:
    """
    Строит COCO-словарь из обработанных результатов.

    Структура:
        {
            "images": [{"id": N, "file_name": "...", "width": W, "height": H}, ...],
            "annotations": [{"id": M, "image_id": N, "category_id": C, "bbox": [...],
                             "segmentation": [...], "area": A, "iscrowd": 0}, ...],
            "categories": [...]
        }
    """
    images = []
    annotations = []
    ann_id = 0

    for img_id, result in enumerate(results):
        pdf_name = result["pdf_name"]
        images.append({
            "id": img_id,
            "file_name": pdf_name,
            "width": result["width"],
            "height": result["height"],
        })

        for region in result["regions"]:
            bbox = region["bbox"]  # [x, y, w, h]
            area = bbox[2] * bbox[3]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": region["category_id"],
                "bbox": bbox,
                "segmentation": [bbox],  # RLE или полигон; здесь — копия bbox
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Генерация val.json с идеальными регионами из полносвязного графа",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pdf-dir", required=True,
        help="Директория с PDF-файлами (напр. .../pdfs_val)",
    )
    parser.add_argument(
        "--coco", required=True,
        help="Путь к исходному COCO JSON (напр. val.json)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Куда сохранить результат (напр. val_ideal.json)",
    )
    parser.add_argument(
        "--dataset-name", default="publaynet",
        choices=["publaynet", "doclaynet"],
        help="Имя датасета (влияет на коэффициент масштабирования координат)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Число параллельных процессов (по умолчанию min(cpu_count, 4))",
    )
    args = parser.parse_args()

    pdf_dir = args.pdf_dir
    coco_path = args.coco
    output_path = args.output
    dataset_name = args.dataset_name
    workers = args.workers or min(cpu_count(), 4)

    # ── Загружаем категории из COCO (нужны для финального JSON) ──
    with open(coco_path, "r") as f:
        coco_meta = json.load(f)
    categories = coco_meta.get("categories", [])

    # ── Собираем список PDF ──
    pdf_names = _collect_pdf_names(pdf_dir)
    total = len(pdf_names)
    print(f"PDF directory : {pdf_dir}")
    print(f"COCO source   : {coco_path}")
    print(f"Total PDFs    : {total}")
    print(f"Workers       : {workers}")
    print(f"Output        : {output_path}")
    print()

    # ── Параллельная обработка ──
    results: List[dict] = []
    ctx = multiprocessing.get_context("spawn")

    try:
        from tqdm import tqdm
        with ctx.Pool(
            workers,
            initializer=_init_worker,
            initargs=(pdf_dir, coco_path, dataset_name),
        ) as pool:
            for item in tqdm(
                pool.imap_unordered(_process_one, pdf_names),
                total=total,
            ):
                if item is not None:
                    results.append(item)
    except ImportError:
        with ctx.Pool(
            workers,
            initializer=_init_worker,
            initargs=(pdf_dir, coco_path, dataset_name),
        ) as pool:
            for item in pool.imap_unordered(_process_one, pdf_names):
                if item is not None:
                    results.append(item)

    # ── Сортируем для детерминированного порядка ──
    results.sort(key=lambda r: r["pdf_name"])

    n_processed = len(results)
    n_regions = sum(len(r["regions"]) for r in results)
    print(f"\nProcessed: {n_processed}/{total} PDFs")
    print(f"Total ideal regions: {n_regions}")

    # ── Строим итоговый COCO ──
    coco_out = _build_coco(results, categories)

    with open(output_path, "w") as f:
        json.dump(coco_out, f, ensure_ascii=False)

    print(f"Saved: {output_path}")
    print(f"  images     : {len(coco_out['images'])}")
    print(f"  annotations: {len(coco_out['annotations'])}")
    print(f"  categories : {len(coco_out['categories'])}")


if __name__ == "__main__":
    main()