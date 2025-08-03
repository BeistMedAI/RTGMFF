from typing import List, Tuple, Optional
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score


def compute_f1(labels: List[int], preds: List[int], average: str = 'macro') -> float:
    return f1_score(labels, preds, average=average)


def compute_balanced_accuracy(labels: List[int], preds: List[int]) -> float:
    return balanced_accuracy_score(labels, preds)


def compute_confusion_matrix(labels: List[int], preds: List[int], num_classes: Optional[int] = None) -> np.ndarray:
    return confusion_matrix(labels, preds, labels=list(range(num_classes)) if num_classes is not None else None)