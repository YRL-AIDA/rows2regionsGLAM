import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from pagerlib.dtypes import ImageSegment


COLORS_CLASS = [
    'gray',
    'tab:red',
    'tab:green',
    'tab:blue',
    'tab:orange',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:olive',
    'tab:cyan',
    'lime',
    'teal',
]
NAME2INT = {
    'other':0, 'text': 1, 'title': 2, 'list': 3, 'table': 4, 'figure':5 
}

class PageRender:
    def __init__(self, id2name=None):
        self.id2name = id2name or {}

    def render(
        self,
        pdf_json,
        img,
        torch_dict,
        true_regions,
        true_categories,
        pred_regions=None,
        pred_categories=None,
        metrics=None,
        save_path=None,
        dpi=150,
        title=None,
    ):
        rows = pdf_json.get("rows", [])
        page_w = pdf_json.get("width", img.shape[1])
        page_h = pdf_json.get("height", img.shape[0])

        fig, ax = plt.subplots(figsize=(page_w / 100, page_h / 100), dpi=dpi)
        ax.imshow(img, extent=[0, page_w, page_h, 0])

        self._draw_rows(ax, rows)
        print(true_categories, pred_categories)
        pred_categories = [NAME2INT[p] for p in pred_categories]
        self._draw_true_regions(ax, true_regions, true_categories)
        self._draw_pred_regions(ax, pred_regions, pred_categories)
        self._draw_graph(ax, torch_dict, rows)

        if metrics:
            self._draw_metrics(ax, metrics)

        if title:
            ax.set_title(title, fontsize=10)

        ax.set_xlim(0, page_w)
        ax.set_ylim(page_h, 0)
        ax.axis("off")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            return save_path
        else:
            plt.show()
            return fig

    def _draw_rows(self, ax, rows):
        for row in rows:
            seg = row.get("segment", {})
            if not seg:
                continue
            x0 = seg.get("x_top_left", 0)
            y0 = seg.get("y_top_left", 0)
            w = seg.get("width", 0)
            h = seg.get("height", 0)
            ax.plot(
                [x0, x0 + w, x0 + w, x0, x0],
                [y0, y0, y0 + h, y0 + h, y0],
                color="lightgray",
                linewidth=0.3,
            )

    def _draw_true_regions(self, ax, regions, categories):
        if not regions:
            return
        for reg, cat in zip(regions, categories):
            color = COLORS_CLASS[int(cat) % len(COLORS_CLASS)]
            name = self.id2name.get(int(cat), str(cat))
            x0 = reg.x_top_left
            y0 = reg.y_top_left
            x1 = reg.x_bottom_right
            y1 = reg.y_bottom_right
            ax.plot(
                [x0, x0, x1, x1, x0],
                [y0, y1, y1, y0, y0],
                color=color,
                linewidth=1.5,
                linestyle="-",
            )
            ax.text(x0 + 2, y0 + 10, name, color=color, fontsize=5, fontweight="bold")

    def _draw_pred_regions(self, ax, regions, categories):
        if not regions:
            return
        for reg, cat in zip(regions, categories):
            color = COLORS_CLASS[int(cat) % len(COLORS_CLASS)]
            name = self.id2name.get(int(cat), str(cat))
            seg = reg.get("segment", reg) if isinstance(reg, dict) else reg
            if isinstance(seg, dict):
                x0 = seg.get("x_top_left", 0)
                y0 = seg.get("y_top_left", 0)
                x1 = x0 + seg.get("width", 0)
                y1 = y0 + seg.get("height", 0)
            else:
                x0 = seg.x_top_left
                y0 = seg.y_top_left
                x1 = seg.x_bottom_right
                y1 = seg.y_bottom_right
            ax.plot(
                [x0, x0, x1, x1, x0],
                [y0, y1, y1, y0, y0],
                color=color,
                linewidth=1.0,
                linestyle="--",
            )
            ax.text(x1 - 30, y1 - 4, name, color=color, fontsize=5, fontweight="bold")

    def _draw_graph(self, ax, torch_dict, rows):
        A = torch_dict.get("inds", [[], []])
        X = torch_dict.get("X", None)
        if X is None or len(A[0]) == 0:
            return

        true_edges = torch_dict.get("true_edges", [True] * len(A[0]))

        for idx, (i, j) in enumerate(zip(A[0], A[1])):
            is_edge = bool(true_edges[idx]) if idx < len(true_edges) else True
            color = "green" if is_edge else "red"
            seg_i = rows[i].get("segment", {})
            seg_j = rows[j].get("segment", {})
            cx_i = seg_i.get("x_top_left", 0) + seg_i.get("width", 0) / 2
            cy_i = seg_i.get("y_top_left", 0) + seg_i.get("height", 0) / 2
            cx_j = seg_j.get("x_top_left", 0) + seg_j.get("width", 0) / 2
            cy_j = seg_j.get("y_top_left", 0) + seg_j.get("height", 0) / 2
            ax.plot([cx_i, cx_j], [cy_i, cy_j], color=color, linewidth=0.4, alpha=0.7)

    def _draw_metrics(self, ax, metrics):
        text_lines = []
        for key, val in metrics.items():
            if isinstance(val, float):
                text_lines.append(f"{key}: {val:.4f}")
            else:
                text_lines.append(f"{key}: {val}")
        text_str = "\n".join(text_lines)
        ax.text(
            0.02,
            0.98,
            text_str,
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            family="monospace",
        )


def render_page_from_pipeline(
    rez,
    pdf_json,
    img,
    true_regions,
    true_categories,
    save_dir,
    page_name,
    metrics=None,
    id2name=None,
):
    renderer = PageRender(id2name=id2name)

    torch_dict = rez.get("torch_dict", {})
    regions_pred = rez.get("regions", [])

    pred_regions = []
    pred_categories = []
    for r in regions_pred:
        seg = r.get("segment", r)
        if isinstance(seg, dict):
            pred_regions.append(
                ImageSegment(
                    seg.get("x_top_left", 0),
                    seg.get("y_top_left", 0),
                    seg.get("x_top_left", 0) + seg.get("width", 0),
                    seg.get("y_top_left", 0) + seg.get("height", 0),
                )
            )
        else:
            pred_regions.append(seg)
        label = r.get("label", r.get("data", {}).get("label", 0))
        pred_categories.append(label)

    save_path = Path(save_dir) / f"{Path(page_name).stem}.png"
    renderer.render(
        pdf_json=pdf_json,
        img=img,
        torch_dict=torch_dict,
        true_regions=true_regions,
        true_categories=true_categories,
        pred_regions=pred_regions,
        pred_categories=pred_categories,
        metrics=metrics,
        save_path=str(save_path),
        title=page_name,
    )
    return save_path
