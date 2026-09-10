import numpy as np
from sklearn.metrics import precision_recall_curve
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].numpy()

    precision, recall, thresholds = precision_recall_curve(labels, probs)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)

    best_f1 = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]

    preds_05 = np.argmax(logits.numpy(), axis=-1)
    acc = (preds_05 == labels).mean()

    return {
        "f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "accuracy": float(acc)
    }