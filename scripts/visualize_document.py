#!/usr/bin/env python3
"""
Visualize a PDF document page with predicted and ground truth regions overlaid.

Loads a trained GLAM model (Rows2Regions GNN), processes a PDF document page,
runs inference to predict region segmentation and classification, and renders
the page with predicted regions (dashed), ground truth regions (solid),
GNN graph edges, and row boundaries.

Example usage::

    python scripts/visualize_document.py \
        --config config.json \
        --model result/row2region_GLAM_no_font_seed_0 \
        --pdf /path/to/document.pdf \
        --output output.png \
        --coco /path/to/val.json \
        --tokenizer no_font \
        --dataset publaynet \
        --dpi 150 \
        --page 0

Device selection: set ``DEVICE=cpu`` or ``DEVICE=cuda`` in the project ``.env``
file, or export the ``DEVICE`` environment variable.  Falls back to CPU if CUDA
is unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup – add the project root so that `rows2regionsGLAM` is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visualize_document")


# ===================================================================
# Helpers
# ===================================================================


def _get_device() -> torch.device:
    """Determine the torch device from the project ``.env`` or environment.

    Priority:
    1. ``DEVICE`` key in ``$PROJECT_ROOT/.env`` (via python-dotenv).
    2. ``DEVICE`` environment variable.
    3. Fallback to ``cpu``.

    If ``cuda`` is requested but not available, falls back to ``cpu`` with a
    warning.
    """
    env_path = _PROJECT_ROOT / ".env"
    device_str = "cpu"

    if env_path.exists():
        try:
            from dotenv import dotenv_values

            env_vars = dotenv_values(env_path)
            device_str = env_vars.get("DEVICE", "cpu")
        except ImportError:
            logger.warning("python-dotenv not installed; reading DEVICE from os.environ")
            device_str = os.environ.get("DEVICE", "cpu")
    else:
        device_str = os.environ.get("DEVICE", "cpu")

    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        device_str = "cpu"

    return torch.device(device_str)


def _move_tensor_dict(d: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Recursively move every ``torch.Tensor`` value in *d* to *device*."""
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, (list, tuple)):
            result[k] = type(v)(
                _move_tensor_dict({"_": x}, device)["_"] if isinstance(x, torch.Tensor) else x
                for x in v
            )
        else:
            result[k] = v
    return result


class _DeviceAwareModel:
    """Thin wrapper around a nn.Module that moves the tokenizer's CPU tensors
    to the model's device before every forward pass.

    Needed because :class:`~rows2regionsGLAM.pipeline.converters.Rows2Regions`
    calls the tokenizer internally (producing CPU tensors) and then immediately
    calls the model, which may reside on GPU.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self._model = model
        self._device = device

    def __call__(self, graph_dict: Dict[str, Any]) -> Dict[str, Any]:
        device_dict = _move_tensor_dict(graph_dict, self._device)
        return self._model(device_dict)

    def eval(self) -> None:
        self._model.eval()

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute access to the wrapped model for transparency.
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self._model, name)


# ===================================================================
# Config loading
# ===================================================================


def load_config(config_path: str) -> dict:
    """Load model architecture parameters from a JSON file.

    The JSON must contain at minimum ``node_block``, ``node_classifier_block``,
    and ``edge_classifier_block`` (matching the output of
    :func:`~experiments.core.model_params.default_arch`).  Extra keys such as
    ``_tokenizer``, ``_input_dim``, ``epochs``, ``batch_size``, etc. are
    preserved but ignored by the model constructor.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        KeyError: If required architecture keys are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    required = ["node_classifier_block", "edge_classifier_block"]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(
            f"Config file {config_path} is missing required keys: {missing}. "
            f"Expected at least: {required}"
        )

    return config


def set_inference_activations(params: dict) -> dict:
    """Replace training activations with inference-mode activations.

    - Node classifier: last layer → ``"softmax"``
    - Edge classifier: last layer → ``"sigmoid"``
    """
    # Make a shallow copy so we don't mutate the caller's dict.
    params = dict(params)

    nc_block = dict(params["node_classifier_block"])
    nc_post = list(nc_block["linear_post"])
    nc_post[-1] = dict(nc_post[-1])
    nc_post[-1]["activation"] = "softmax"
    nc_block["linear_post"] = nc_post
    params["node_classifier_block"] = nc_block

    ec_block = dict(params["edge_classifier_block"])
    ec_post = list(ec_block["linear_post"])
    ec_post[-1] = dict(ec_post[-1])
    ec_post[-1]["activation"] = "sigmoid"
    ec_block["linear_post"] = ec_post
    params["edge_classifier_block"] = ec_block

    return params


# ===================================================================
# Tokenizer factory
# ===================================================================


def create_tokenizer(tokenizer_type: str):
    """Instantiate the tokenizer corresponding to *tokenizer_type*.

    Supported values:

    - ``"no_font"`` – baseline 21-dim tokenizer (no font features).
    - ``"pdf_font"`` – 24-dim tokenizer with 3 PDF font features.
    - ``"font_emb_16"`` / ``"font_emb_32"`` / ``"font_emb_512"`` – font
      embedding tokenizer with the specified embedding dimension.
    """
    if tokenizer_type == "no_font":
        from rows2regionsGLAM.tokenizers.base_line_tokenizer import RowGLAMTokenizer

        return RowGLAMTokenizer()

    if tokenizer_type == "pdf_font":
        from rows2regionsGLAM.tokenizers.font_tokenizer import RowGLAMTokenizer

        return RowGLAMTokenizer()

    if tokenizer_type.startswith("font_emb_"):
        size_str = tokenizer_type.split("_")[-1]
        try:
            size = int(size_str)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse embedding size from tokenizer type '{tokenizer_type}'"
            ) from exc

        from rows2regionsGLAM.tokenizers.font_emb_tokenizer import RowGLAMTokenizer

        return RowGLAMTokenizer(size=size)

    raise ValueError(
        f"Unknown tokenizer type: '{tokenizer_type}'. "
        f"Expected one of: no_font, pdf_font, font_emb_16, font_emb_32, font_emb_512"
    )


# ===================================================================
# COCO ground-truth loading
# ===================================================================


def _load_ground_truth(
    coco_path: Optional[str],
    dataset: str,
    pdf_name: str,
    pdf_json: dict,
) -> Tuple[List, List, dict]:
    """Load ground truth regions and categories from a COCO annotation file.

    Args:
        coco_path: Path to the COCO JSON, or ``None``.
        dataset: ``"publaynet"`` or ``"doclaynet"``.
        pdf_name: Filename of the PDF (used as key in COCO annotations).
        pdf_json: Page info dict (``{"width": ..., "height": ...}``) used for
            coordinate scaling in DocLayNet mode.

    Returns:
        Tuple of ``(regions, categories, id2name)``.  Regions is a list of
        :class:`~pagerlib.dtypes.ImageSegment`, categories a list of
        ``int`` category IDs, and *id2name* a ``{id: name}`` mapping.
    """
    if not coco_path:
        # Default PubLayNet classes when no COCO is provided.
        if dataset == "publaynet":
            id2name: dict = {0: "other", 1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}
        else:
            id2name = {}
        return [], [], id2name

    from rows2regionsGLAM.utils.coco_manager import COCOManager

    coco = COCOManager(coco_path=coco_path, name_dataset=dataset)

    try:
        regions, categories = coco(pdf_name, pdf_json)
    except KeyError:
        logger.warning(
            "No ground-truth annotations found for '%s' in %s", pdf_name, coco_path
        )
        return [], [], coco.coco_classes

    return regions, categories, coco.coco_classes


# ===================================================================
# Inference via Rows2Regions converter
# ===================================================================


def _run_inference(
    model: torch.nn.Module,
    tokenizer,
    pdf_json: dict,
    img: np.ndarray,
    device: torch.device,
    id2name: dict,
) -> Tuple[List, List]:
    """Run the GLAM model on a single page and return predicted regions.

    Uses the :class:`~rows2regionsGLAM.pipeline.converters.Rows2Regions`
    converter, which performs GNN inference followed by connected-component
    extraction on the surviving edges.

    Args:
        model: Trained GLAM model (``TorchModel``).
        tokenizer: Tokenizer instance that produces the graph dict.
        pdf_json: Page structure dict (``{"rows": [...], "width": W, "height": H}``).
        img: Page image as a numpy array (H×W×3, RGB).
        device: torch device for inference.
        id2name: Category-ID-to-name mapping (used for label assignment).

    Returns:
        Tuple of ``(pred_segments, pred_categories)`` where *pred_segments* is
        a list of :class:`~pagerlib.dtypes.ImageSegment` and
        *pred_categories* a list of category labels (int or str).
    """
    from rows2regionsGLAM._page_model import RegionModel, RowsModel
    from rows2regionsGLAM.pipeline.converters import Rows2Regions

    device_aware = _DeviceAwareModel(model, device)

    converter = Rows2Regions(
        {
            "model": device_aware,
            "tokenizer": tokenizer,
            "is_merge_extract": True,
            "classes": id2name,
        }
    )

    rows_model = RowsModel()
    rows_model.from_dict({"rows": pdf_json["rows"]})
    region_model = RegionModel()
    converter.convert(rows_model, region_model, img)

    pred_region_objs = region_model.to_dict()["regions"]
    pred_segments = [r.segment for r in pred_region_objs]
    pred_categories = [r.data.get("label", 0) for r in pred_region_objs]

    return pred_segments, pred_categories


# ===================================================================
# Main entry point
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize document page with predicted and ground truth regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Required arguments ---
    parser.add_argument(
        "--config",
        required=True,
        help="JSON file with model architecture params (e.g. from default_arch())",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model checkpoint (file or directory prefix)",
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF document to visualize")
    parser.add_argument(
        "--output", required=True, help="Output image path (PNG, PDF, SVG, etc.)"
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        choices=["no_font", "pdf_font", "font_emb_16", "font_emb_32", "font_emb_512"],
        help="Tokenizer type",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["publaynet", "doclaynet"],
        help="Dataset name (affects COCO coordinate scaling)",
    )

    # --- Optional arguments ---
    parser.add_argument(
        "--coco",
        default=None,
        help="Path to COCO annotations JSON for ground truth (optional)",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Skip ground truth rendering even if --coco is provided",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip drawing GNN graph edges between rows",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI (default: 150)",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="Zero-based page number to render (default: 0)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Device
    # ------------------------------------------------------------------
    device = _get_device()
    logger.info("Using device: %s", device)
    logger.info(
        "Tip: set DEVICE=cpu or DEVICE=cuda in %s/.env or via environment variable",
        _PROJECT_ROOT,
    )

    # ------------------------------------------------------------------
    # 2. Load model config
    # ------------------------------------------------------------------
    params = load_config(args.config)
    params = set_inference_activations(params)
    logger.info("Loaded config from: %s", args.config)

    # ------------------------------------------------------------------
    # 3. Load trained model
    # ------------------------------------------------------------------
    from rows2regionsGLAM.models import get_model

    model, _ = get_model(params, args.model)
    model.to(device)
    model.eval()
    logger.info("Loaded model from: %s", args.model)

    # ------------------------------------------------------------------
    # 4. Create tokenizer
    # ------------------------------------------------------------------
    tokenizer = create_tokenizer(args.tokenizer)
    logger.info("Created tokenizer: %s", args.tokenizer)

    # ------------------------------------------------------------------
    # 5. Process PDF (page image + row structure)
    # ------------------------------------------------------------------
    from rows2regionsGLAM.utils.pdf_manager import PDFManager
    from rows2regionsGLAM.utils.row_manager import RowManager

    pdf_manager = PDFManager()
    row_manager = RowManager()

    raw_json, img = pdf_manager.get_json_and_img_from_pdf(args.pdf, num_page=args.page)
    if img is None:
        logger.error(
            "Failed to extract page %d from '%s'. The PDF may be corrupt or "
            "the page number is out of range.",
            args.page,
            args.pdf,
        )
        sys.exit(1)

    row_json = row_manager.get_row_json_from_pdf_json(raw_json)
    pdf_json: dict = {
        "rows": row_json,
        "width": raw_json["width"],
        "height": raw_json["height"],
    }

    logger.info(
        "Processed PDF: %s (page %d) – %d rows, %d×%d px",
        args.pdf,
        args.page,
        len(pdf_json["rows"]),
        pdf_json["width"],
        pdf_json["height"],
    )

    # ------------------------------------------------------------------
    # 6. Ground truth (COCO)
    # ------------------------------------------------------------------
    pdf_name = Path(args.pdf).name

    if args.no_ground_truth:
        if args.dataset == "publaynet":
            id2name = {0: "other", 1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}
        else:
            id2name = {}
        true_regions: list = []
        true_categories: list = []
        logger.info("Ground truth rendering disabled via --no-ground-truth")
    else:
        true_regions, true_categories, id2name = _load_ground_truth(
            args.coco, args.dataset, pdf_name, pdf_json
        )
        logger.info("Loaded ground truth: %d regions", len(true_regions))

    # ------------------------------------------------------------------
    # 7. Run inference
    # ------------------------------------------------------------------
    pred_segments, pred_categories = _run_inference(
        model=model,
        tokenizer=tokenizer,
        pdf_json=pdf_json,
        img=img,
        device=device,
        id2name=id2name,
    )
    logger.info("Predicted %d regions", len(pred_segments))

    # ------------------------------------------------------------------
    # 8. Tokenize for graph rendering (edges between rows)
    # ------------------------------------------------------------------
    if not args.no_graph:
        torch_dict = tokenizer(pdf_json["rows"], img)
        # Move all tensors back to CPU – matplotlib only runs on CPU.
        torch_dict = _move_tensor_dict(torch_dict, torch.device("cpu"))
    else:
        torch_dict = {}

    # ------------------------------------------------------------------
    # 9. Render with PageRender
    # ------------------------------------------------------------------
    from rows2regionsGLAM.utils.ploter.page_render import PageRender

    renderer = PageRender(id2name=id2name)
    output_path = renderer.render(
        pdf_json=pdf_json,
        img=img,
        torch_dict=torch_dict,
        true_regions=true_regions,
        true_categories=true_categories,
        pred_regions=pred_segments,
        pred_categories=pred_categories,
        save_path=args.output,
        dpi=args.dpi,
        title=Path(args.pdf).stem,
    )
    logger.info("Saved visualization to: %s", output_path)


if __name__ == "__main__":
    main()
