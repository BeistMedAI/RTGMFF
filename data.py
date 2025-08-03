# -*- coding: utf-8 -*-
"""
Data handling utilities for the RTGMFF reproduction.

This module defines dataset classes and helper functions used to
prepare inputs for the RTGMFF model.  It implements the
FMRIImageDataset for loading preprocessed fMRI slices along with
demographic information, and the ROITextGenerator for converting
per‑region activation vectors into discretised token sequences.  It also
contains utilities for building a token vocabulary and converting
activation vectors into integer indices.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class FMRIImageDataset(Dataset):
    """Dataset for 2‑D fMRI representations and metadata.

    Each sample returns a dictionary containing the 3‑channel image
    (ALFF/fALFF/ReHo), the diagnosis label, age, gender and an optional
    site identifier.  The ROI indices (generated separately) can be
    attached to the dataset by assigning an array to the `roi_indices`
    attribute after initialisation.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, H, W, 3) with per‑subject feature maps.
    labels : np.ndarray
        Diagnosis labels for each subject.
    ages : np.ndarray
        Subject ages (floats).
    genders : np.ndarray
        Gender encodings (0=female, 1=male).
    sites : Optional[np.ndarray]
        Optional vector specifying imaging site for each subject.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 ages: np.ndarray, genders: np.ndarray,
                 sites: Optional[np.ndarray] = None) -> None:
        assert images.ndim == 4 and images.shape[-1] == 3, "images must be N×H×W×3"
        assert labels.shape[0] == images.shape[0]
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.ages = ages.astype(np.float32)
        self.genders = genders.astype(np.int64)
        self.sites = sites.astype(np.int64) if sites is not None else None
        # Placeholder for ROI indices; to be assigned externally
        self.roi_indices: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Convert image from H×W×C to C×H×W
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        gender = torch.tensor(self.genders[idx], dtype=torch.long)
        sample = {
            'image': image,
            'label': label,
            'age': age,
            'gender': gender,
        }
        if self.sites is not None:
            sample['site'] = torch.tensor(self.sites[idx], dtype=torch.long)
        if self.roi_indices is not None:
            sample['roi_indices'] = torch.tensor(self.roi_indices[idx], dtype=torch.long)
        return sample

def load_nifti_volume(path: str) -> np.ndarray:
    """Load a 4‑D fMRI volume from a NIfTI file.

    This helper attempts to import the `nibabel` library to read
    NIfTI files.  If `nibabel` is not installed an ImportError will
    be raised.  The returned array has shape (X, Y, Z, T) and
    dtype float32.

    Parameters
    ----------
    path : str
        Path to a .nii or .nii.gz file on disk.

    Returns
    -------
    np.ndarray
        The 4‑D data as a numpy array.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError(
            "nibabel is required to load NIfTI files. Please install it via pip install nibabel"
        ) from e
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data


def compute_alff_falff_reho(bold_ts: np.ndarray, fs: float = 1.0,
                            low_freq: float = 0.01, high_freq: float = 0.08) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ALFF, fALFF and ReHo for a 4‑D BOLD time series.

    This function provides an approximate implementation of the
    frequency‑domain metrics used in the paper【569137259789883†screenshot】.  It computes the
    amplitude of low‑frequency fluctuations (ALFF), fractional ALFF
    (fALFF) and regional homogeneity (ReHo) for each voxel.  The
    formulas here are simplified for clarity and may not match
    specialised neuroimaging toolboxes exactly.

    Parameters
    ----------
    bold_ts : np.ndarray
        4‑D array of shape (X, Y, Z, T) containing the BOLD signal.
    fs : float
        Sampling frequency (Hz) of the time series.
    low_freq, high_freq : float
        Frequency band (Hz) for the ALFF/fALFF computation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Three 3‑D arrays (X, Y, Z) containing ALFF, fALFF and ReHo
        values respectively.
    """
    X, Y, Z, T = bold_ts.shape
    # Compute power spectrum via FFT
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    fft_vals = np.fft.rfft(bold_ts, axis=-1)
    power = np.abs(fft_vals) ** 2
    # Identify indices for low‑frequency band and full band
    band_idx = (freqs >= low_freq) & (freqs <= high_freq)
    # ALFF: root mean square of power within low frequency band
    alff = np.sqrt(power[..., band_idx].mean(axis=-1))
    # Total power for fALFF
    total_power = power.mean(axis=-1)
    falff = alff / (np.sqrt(total_power) + 1e-8)
    # ReHo: compute Kendall's coefficient of concordance over neighbours
    # For simplicity we compute voxel‑wise standard deviation of the time
    # series as a proxy for local synchrony.  A proper implementation
    # would require computing concordance across a 27‑voxel neighbourhood.
    std_map = bold_ts.std(axis=-1)
    reho = 1.0 / (std_map + 1e-8)
    return alff, falff, reho


def bold_to_2d_representation(bold_ts: np.ndarray, atlas_labels: np.ndarray,
                              layout: Tuple[int, int] = (116, 1)) -> np.ndarray:
    """Convert 4‑D BOLD to a 2‑D 3‑channel representation.

    The paper reduces the 4‑D fMRI volume to a 2‑D layout of size
    H×W with three channels corresponding to ALFF, fALFF and ReHo【569137259789883†screenshot】.  This
    helper takes a BOLD time series and a parcellation label map and
    produces a 2‑D feature map.  The default layout simply stacks
    regions vertically; users may customise the layout by specifying
    the `layout` parameter or replacing this function.

    Parameters
    ----------
    bold_ts : np.ndarray
        4‑D BOLD time series (X, Y, Z, T).
    atlas_labels : np.ndarray
        3‑D array of atlas labels (X, Y, Z) assigning each voxel to an
        ROI index from 1 to 116.  Voxels with label 0 are ignored.
    layout : Tuple[int, int]
        Desired arrangement of the 116 regions; default (116, 1) places
        each ROI on a new row.

    Returns
    -------
    np.ndarray
        3‑channel image of shape (H, W, 3) combining ALFF, fALFF and
        ReHo per region.  Each channel is normalised to [0,1].
    """
    alff, falff, reho = compute_alff_falff_reho(bold_ts)
    H, W = layout
    assert H * W == 116, "layout must accommodate 116 regions"
    img = np.zeros((H, W, 3), dtype=np.float32)
    # For each ROI compute mean metric across voxels
    for roi_idx in range(1, 117):
        mask = atlas_labels == roi_idx
        if not np.any(mask):
            continue
        row = (roi_idx - 1) // W
        col = (roi_idx - 1) % W
        img[row, col, 0] = alff[mask].mean()
        img[row, col, 1] = falff[mask].mean()
        img[row, col, 2] = reho[mask].mean()
    # Normalise channels
    for c in range(3):
        channel = img[..., c]
        if channel.max() > channel.min():
            img[..., c] = (channel - channel.min()) / (channel.max() - channel.min())
    return img


class ROITextGenerator:
    """Generates discretised ROI activation tokens.

    A deterministic rule‑based generator that maps a 116‑dimensional
    activation vector into triplets (ROI, strength, polarity) using two
    thresholds τ₁ and τ₂.  The strength category is weak if |vᵢ| < τ₁,
    moderate if τ₁ ≤ |vᵢ| < τ₂, and strong otherwise, while polarity
    indicates the sign (↑ for positive, ↓ for negative)【569137259789883†screenshot】.
    """

    ROI_NAMES: List[str] = [f'ROI_{i+1:03d}' for i in range(116)]

    def __init__(self, tau1: float = 0.15, tau2: float = 0.30) -> None:
        assert tau1 < tau2, "tau1 must be less than tau2"
        self.tau1 = tau1
        self.tau2 = tau2

    def discretise_vector(self, v: np.ndarray) -> List[Tuple[str, str, str]]:
        tokens: List[Tuple[str, str, str]] = []
        for roi_name, val in zip(self.ROI_NAMES, v):
            magnitude = abs(val)
            if magnitude >= self.tau2:
                strength = 'strong'
            elif magnitude >= self.tau1:
                strength = 'moderate'
            else:
                strength = 'weak'
            polarity = '↑' if val >= 0 else '↓'
            tokens.append((roi_name, strength, polarity))
        return tokens

    def tokens_to_indices(self, tokens: List[Tuple[str, str, str]], vocab: Dict[str, int]) -> List[int]:
        indices: List[int] = []
        for roi_name, strength, polarity in tokens:
            token_str = f'{roi_name}/{strength}/{polarity}'
            if token_str not in vocab:
                raise KeyError(f"Token {token_str} not in vocabulary")
            indices.append(vocab[token_str])
        return indices


def build_vocab() -> Dict[str, int]:
    """Create a full vocabulary for all ROI tokens.

    The vocabulary maps each possible (ROI, strength, polarity) triplet
    string to a unique integer ID.  There are 116 × 3 × 2 possible
    combinations.
    """
    vocab: Dict[str, int] = {}
    idx = 0
    for roi_i in range(116):
        roi_name = f'ROI_{roi_i+1:03d}'
        for strength in ['weak', 'moderate', 'strong']:
            for polarity in ['↑', '↓']:
                tok = f'{roi_name}/{strength}/{polarity}'
                vocab[tok] = idx
                idx += 1
    return vocab


def prepare_roi_indices(activation_vectors: np.ndarray, tau1: float, tau2: float,
                        vocab: Dict[str, int]) -> np.ndarray:
    """Discretise activation vectors and map them to integer indices.

    Parameters
    ----------
    activation_vectors : np.ndarray
        Array of shape (N, 116) with ΔBOLD values per subject.
    tau1, tau2 : float
        Thresholds for discretisation.
    vocab : Dict[str, int]
        Mapping from token strings to indices.

    Returns
    -------
    np.ndarray
        Array of shape (N, 116) containing integer token IDs for each
        subject.
    """
    roi_gen = ROITextGenerator(tau1, tau2)
    all_indices: List[List[int]] = []
    for vec in activation_vectors:
        tokens = roi_gen.discretise_vector(vec)
        indices = []
        for tok in tokens:
            token_str = f'{tok[0]}/{tok[1]}/{tok[2]}'
            indices.append(vocab[token_str])
        all_indices.append(indices)
    return np.array(all_indices, dtype=np.int64)
