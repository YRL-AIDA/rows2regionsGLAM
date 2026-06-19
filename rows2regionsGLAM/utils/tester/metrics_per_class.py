# ================= IoU-based classification metrics =================
def iou_xywh(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0


def classification_metrics_iou(preds, target, preds_cls, target_cls, thresholds=(0.5, 0.95)):
    from collections import defaultdict

    all_classes = sorted(list(set(c for lst in preds_cls + target_cls for c in lst)))

    def compute_for_threshold(th):
        TP = defaultdict(int)
        FP = defaultdict(int)
        FN = defaultdict(int)

        for b_pred, b_true, c_pred, c_true in zip(preds, target, preds_cls, target_cls):
            used_gt = set()

            for i, pb in enumerate(b_pred):
                pred_class = c_pred[i]
                best_iou = 0
                best_j = -1

                for j, tb in enumerate(b_true):
                    if j in used_gt:
                        continue

                    true_class = c_true[j]
                    if pred_class != true_class:
                        continue

                    val = iou_xywh(pb, tb)
                    if val > best_iou:
                        best_iou = val
                        best_j = j

                if best_iou >= th:
                    TP[pred_class] += 1
                    used_gt.add(best_j)
                else:
                    FP[pred_class] += 1

            for j, gt_class in enumerate(c_true):
                if j not in used_gt:
                    FN[gt_class] += 1

        lines = [f"\nIoU threshold = {th}"]
        for cls in all_classes:
            tp = TP[cls]
            fp = FP[cls]
            fn = FN[cls]

            p = tp / (tp + fp) if tp + fp > 0 else 0
            r = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            lines.append(f"{cls:15s} | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")

        return "\n".join(lines)

    return "\n".join(compute_for_threshold(th) for th in thresholds)


# ================= gridIoU-based classification metrics =================
from rows2regionsGLAM.metrics.grid_metrics import gridIoU


def classification_metrics_grid(preds, target, preds_cls, target_cls, row_grids, thresholds=(0.5, 0.95), dict_classes=None):
    from collections import defaultdict
    from pagerlib.dtypes import ImageSegment

    all_classes = [id_ for id_, name in dict_classes.items()]

    def to_segments(boxes):
        return [
            ImageSegment(
                x_top_left=b[0],
                y_top_left=b[1],
                x_bottom_right=b[0] + b[2],
                y_bottom_right=b[1] + b[3]
            ) for b in boxes
        ]

    def compute_for_threshold(th):
        TP = defaultdict(int)
        FP = defaultdict(int)
        FN = defaultdict(int)

        for b_pred, b_true, c_pred, c_true, grid in zip(
                preds, target, preds_cls, target_cls, row_grids
        ):
            seg_pred = to_segments(b_pred)
            seg_true = to_segments(b_true)
            seg_grid = to_segments(grid)

            used_gt = set()

            for i, pb in enumerate(seg_pred):
                pred_class = c_pred[i]
                best_score = 0
                best_j = -1

                for j, tb in enumerate(seg_true):
                    if j in used_gt:
                        continue

                    true_class = c_true[j]
                    # ВАЖНО: Сравниваем только с целевыми боксами того же класса
                    if pred_class != true_class:
                        continue

                    val = gridIoU(pb, tb, seg_grid)
                    if val > best_score:
                        best_score = val
                        best_j = j

                if best_score >= th:
                    TP[pred_class] += 1
                    used_gt.add(best_j)
                else:
                    FP[pred_class] += 1

            for j, gt_class in enumerate(c_true):
                if j not in used_gt:
                    FN[gt_class] += 1

        metrics = {}
        for cls in all_classes:
            tp = TP[cls]
            fp = FP[cls]
            fn = FN[cls]

            p = tp / (tp + fp) if tp + fp > 0 else 0
            r = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            metrics[cls] = {
                "P":p,
                "R":r,
                "F1":f1
            }
            

        return metrics

    return {f"{name_m}@IoUGrid[{str(th)}] ({cls if dict_classes is None else dict_classes[int(cls)]})":m  for th in thresholds for cls, metric in compute_for_threshold(th).items() for name_m, m in metric.items()}