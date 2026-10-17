import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_from_predictions(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        raise ValueError("No images were evaluated. Check --dataroot and --classes.")
    if not np.isfinite(y_score).all():
        raise ValueError("Model produced NaN or infinite prediction scores.")
    y_hat = (y_score >= threshold).astype(np.int64)
    result = {
        "count": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "average_precision": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_hat, labels=[0, 1]).tolist(),
    }
    result["auroc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    result["real_accuracy"] = float(accuracy_score(y_true[y_true == 0], y_hat[y_true == 0])) if np.any(y_true == 0) else float("nan")
    result["fake_accuracy"] = float(accuracy_score(y_true[y_true == 1], y_hat[y_true == 1])) if np.any(y_true == 1) else float("nan")
    return result


@torch.no_grad()
def validate(model, opt, transform=None):
    from data import create_dataloader

    loader = create_dataloader(opt)
    model.eval()
    device = next(model.parameters()).device
    y_true, y_score = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if transform is not None:
            images = transform(images)
        scores = model(images).sigmoid().flatten()
        y_score.extend(scores.detach().cpu().tolist())
        y_true.extend(labels.flatten().cpu().tolist())
    result = metrics_from_predictions(y_true, y_score)
    return result["accuracy"], result["average_precision"], result["real_accuracy"], result["fake_accuracy"], np.asarray(y_true), np.asarray(y_score)
