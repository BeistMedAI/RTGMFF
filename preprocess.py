# -*- coding: utf-8 -*-
"""
Preprocessing and threshold search for RTGMFF.

This module encapsulates routines that operate at the dataset level,
including nested cross‑validation for selecting discretisation
thresholds τ₁ and τ₂.  
"""

from typing import List, Tuple
import numpy as np
from sklearn.model_selection import KFold

from sklearn.svm import LinearSVC
from data import ROITextGenerator, build_vocab


def nested_cv_threshold_search(activation_vectors: np.ndarray, labels: np.ndarray,
                              candidate_taus: List[Tuple[float, float]],
                              n_splits_outer: int = 5, n_splits_inner: int = 3) -> Tuple[float, float]:
    """Perform nested cross‑validation to select τ₁ and τ₂.

    Parameters
    ----------
    activation_vectors : np.ndarray
        Array of shape (N, 116) containing ΔBOLD values per subject.
    labels : np.ndarray
        Corresponding diagnosis labels.
    candidate_taus : List[Tuple[float, float]]
        Candidate (τ₁, τ₂) pairs to evaluate.  τ₂ should be at least
        τ₁ + δ for some positive δ.
    n_splits_outer : int
        Number of outer folds.
    n_splits_inner : int
        Number of inner folds.

    Returns
    -------
    Tuple[float, float]
        The (τ₁, τ₂) pair that maximises mean inner‑fold accuracy.
    """
    outer_kf = KFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
    best_tau = (0.0, 0.0)
    best_score = -np.inf
    # Build a full vocabulary once for all candidates
    vocab = build_vocab()
    roi_gen = ROITextGenerator()
    # Outer folds
    for train_val_idx, _ in outer_kf.split(activation_vectors):
        inner_X = activation_vectors[train_val_idx]
        inner_y = labels[train_val_idx]
        inner_kf = KFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
        for tau1, tau2 in candidate_taus:
            if tau2 <= tau1:
                continue
            roi_gen.tau1 = tau1
            roi_gen.tau2 = tau2
            inner_scores = []
            for inner_train_idx, inner_val_idx in inner_kf.split(inner_X):
                X_train = inner_X[inner_train_idx]
                y_train = inner_y[inner_train_idx]
                X_val = inner_X[inner_val_idx]
                y_val = inner_y[inner_val_idx]
                # Bag‑of‑tokens vectorisation
                def vectorise(mat):
                    rows = []
                    for vec in mat:
                        tokens = roi_gen.discretise_vector(vec)
                        bow = np.zeros(len(vocab), dtype=np.float32)
                        for tok in tokens:
                            tok_str = f'{tok[0]}/{tok[1]}/{tok[2]}'
                            bow[vocab[tok_str]] += 1.0
                        rows.append(bow)
                    return np.stack(rows, axis=0)
                X_train_vec = vectorise(X_train)
                X_val_vec = vectorise(X_val)
                # Train linear SVM
                clf = LinearSVC(C=1.0)
                clf.fit(X_train_vec, y_train)
                score = clf.score(X_val_vec, y_val)
                inner_scores.append(score)
            mean_score = float(np.mean(inner_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_tau = (tau1, tau2)
    return best_tau


def search_thresholds(activation_vectors: np.ndarray, labels: np.ndarray,
                      method: str = 'grid',
                      tau1_values: List[float] = None,
                      tau2_values: List[float] = None,
                      random_trials: int = 50,
                      n_splits: int = 5) -> Tuple[float, float]:
    """Convenience wrapper to select τ₁ and τ₂ using different strategies.

    Parameters
    ----------
    activation_vectors : np.ndarray
        ΔBOLD matrix (N, 116).
    labels : np.ndarray
        Target labels.
    method : str
        Selection strategy: 'grid' performs exhaustive search over
        specified `tau1_values` and `tau2_values`; 'random' draws
        random pairs within [0,0.5] and evaluates them.
    tau1_values, tau2_values : List[float]
        Candidate values for grid search.  Ignored if method != 'grid'.
    random_trials : int
        Number of random samples when method == 'random'.
    n_splits : int
        Number of folds for KFold cross validation.

    Returns
    -------
    Tuple[float, float]
        Selected (τ₁, τ₂) pair.
    """
    if method not in ('grid', 'random'):
        raise ValueError("method must be 'grid' or 'random'")
    if method == 'grid':
        if tau1_values is None or tau2_values is None:
            tau1_values = [0.05, 0.10, 0.15, 0.20]
            tau2_values = [0.25, 0.30, 0.35, 0.40]
        candidates = [(t1, t2) for t1 in tau1_values for t2 in tau2_values if t2 > t1]
    else:
        # Random search within [0.05,0.45]
        rng = np.random.default_rng(42)
        candidates = []
        while len(candidates) < random_trials:
            t1 = float(rng.uniform(0.05, 0.45))
            t2 = float(rng.uniform(t1 + 0.02, 0.60))
            candidates.append((t1, t2))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    vocab = build_vocab()
    roi_gen = ROITextGenerator()
    best_pair = (0.0, 0.0)
    best_score = -np.inf
    for tau1, tau2 in candidates:
        if tau2 <= tau1:
            continue
        roi_gen.tau1 = tau1
        roi_gen.tau2 = tau2
        scores: List[float] = []
        for train_idx, val_idx in kf.split(activation_vectors):
            X_train = activation_vectors[train_idx]
            y_train = labels[train_idx]
            X_val = activation_vectors[val_idx]
            y_val = labels[val_idx]
            # Vectorise using bag‑of‑words
            def vectorise(mat):
                rows = []
                for vec in mat:
                    tokens = roi_gen.discretise_vector(vec)
                    bow = np.zeros(len(vocab), dtype=np.float32)
                    for tok in tokens:
                        tok_str = f'{tok[0]}/{tok[1]}/{tok[2]}'
                        bow[vocab[tok_str]] += 1.0
                    rows.append(bow)
                return np.stack(rows, axis=0)
            X_train_vec = vectorise(X_train)
            X_val_vec = vectorise(X_val)
            clf = LinearSVC(C=1.0)
            clf.fit(X_train_vec, y_train)
            scores.append(clf.score(X_val_vec, y_val))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_pair = (tau1, tau2)
    return best_pair
