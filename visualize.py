from typing import List, Dict
import numpy as np
from collections import Counter

from utils.visualization import plot_training_curves, plot_confusion_matrix


def plot_history(history: List[Dict[str, float]], metrics: List[str] = None) -> None:
    """Convenience wrapper to plot training history."""
    plot_training_curves(history, metrics)


def plot_roi_token_distribution(roi_indices: np.ndarray, vocab: Dict[str, int]) -> None:
    """Plot the frequency distribution of ROI tokens.

    Parameters
    ----------
    roi_indices : np.ndarray
        Integer array of shape (N, 116) containing token IDs per subject.
    vocab : Dict[str, int]
        Token vocabulary mapping strings to IDs; this will be inverted
        to obtain names.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('matplotlib is required for plotting.') from e
    # Invert the vocabulary
    inv_vocab = {v: k for k, v in vocab.items()}
    # Flatten indices and count occurrences
    flat = roi_indices.flatten()
    counts = Counter(flat)
    # Sort by frequency
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [inv_vocab[idx] for idx, _ in sorted_items[:20]]
    values = [cnt for _, cnt in sorted_items[:20]]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=90)
    plt.title('Top 20 ROI tokens by frequency')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage of the visualisation functions on synthetic data
    # Generate fake history
    history = [
        {'loss_total': 1.0, 'loss_ce': 0.7, 'loss_align': 0.2},
        {'loss_total': 0.8, 'loss_ce': 0.6, 'loss_align': 0.15},
        {'loss_total': 0.6, 'loss_ce': 0.4, 'loss_align': 0.1},
    ]
    plot_history(history, metrics=['loss_total', 'loss_ce', 'loss_align'])
    # Generate fake ROI distribution
    import random
    from data import build_vocab
    vocab = build_vocab()
    N = 50
    roi_indices = np.random.randint(0, len(vocab), size=(N, 116))
    plot_roi_token_distribution(roi_indices, vocab)