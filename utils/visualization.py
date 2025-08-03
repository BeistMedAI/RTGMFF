from typing import List, Dict
import numpy as np

def plot_training_curves(history: List[Dict[str, float]], metrics: List[str] = None) -> None:
    """Plot training curves for one or more metrics.

    Parameters
    ----------
    history : List[Dict[str, float]]
        A list of dictionaries collected over epochs, each containing
        metric names and values.
    metrics : List[str]
        List of metric keys to plot; if None all keys in the first
        history entry are plotted.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('matplotlib is required for plotting. Please install it.') from e
    if not history:
        print('No history to plot.')
        return
    if metrics is None:
        metrics = list(history[0].keys())
    epochs = np.arange(1, len(history) + 1)
    plt.figure(figsize=(8, 4 * len(metrics)))
    for idx, m in enumerate(metrics):
        values = [h[m] for h in history]
        plt.subplot(len(metrics), 1, idx + 1)
        plt.plot(epochs, values, marker='o')
        plt.title(f'{m} over epochs')
        plt.xlabel('Epoch')
        plt.ylabel(m)
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    """Display a confusion matrix using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        raise ImportError('matplotlib and seaborn are required for plotting.') from e
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()