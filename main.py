from typing import Tuple, List
import os
import sys
import time
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import FMRIImageDataset, build_vocab, prepare_roi_indices, load_nifti_volume, bold_to_2d_representation
from model import RTGMFFModel
from utils.training import train_one_epoch, evaluate, get_loss_function
from log import log_metrics, get_logger
# Ensure local modules (args.py) can be imported when running from outside
sys.path.append(os.path.dirname(__file__))
from args import parse_args, get_model_config, save_args_to_file


def load_metadata_csv(path: str) -> List[dict]:
    """Load participant metadata from a TSV/CSV file.

    The metadata file should contain at least the columns ``subject_id``
    (or ``participant_id``), ``label``, ``age`` and ``gender``.
    ``gender`` is expected to be encoded as 0 for female and 1 for
    male.  Gender codes using characters (e.g. 'F', 'M') will be
    converted automatically.

    Parameters
    ----------
    path : str
        Path to a .tsv or .csv file containing metadata.

    Returns
    -------
    List[dict]
        A list of row dictionaries keyed by column name.
    """
    rows: List[dict] = []
    with open(path, newline='', encoding='utf-8') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            rows.append(row)
    return rows


def load_dataset(dataset_root: str, atlas_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load fMRI data and corresponding metadata from a given dataset directory.

    This helper expects the following structure::

        dataset_root/
          participants.tsv  or phenotype.csv
          sub_<id>/
            bold.nii or bold.nii.gz
            ...

    The atlas file should be a NIfTI volume with integer labels from
    1–116 indicating the AAL regions.  For each subject, the BOLD
    volume is loaded using ``load_nifti_volume``, converted into a
    2‑D representation via ``bold_to_2d_representation``, and an
    activation vector is computed by averaging the time course within
    each ROI.

    If either the dataset directory or the atlas cannot be found,
    synthetic data will be generated as a fallback.

    Parameters
    ----------
    dataset_root : str
        Root directory of the dataset.
    atlas_path : str
        Path to the atlas NIfTI file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Arrays of images (N,H,W,3), labels (N,), ages (N,), genders (N,),
        and activation vectors (N,116).
    """
    # Validate inputs
    if not (dataset_root and os.path.exists(dataset_root) and atlas_path and os.path.exists(atlas_path)):
        # Fallback: generate a small synthetic dataset as in the original script
        num_subjects = 16
        img_height, img_width = 64, 64
        images = np.random.rand(num_subjects, img_height, img_width, 3).astype(np.float32)
        labels = np.random.randint(0, 2, size=(num_subjects,), dtype=np.int64)
        ages = np.random.uniform(6, 60, size=(num_subjects,)).astype(np.float32)
        genders = np.random.randint(0, 2, size=(num_subjects,), dtype=np.int64)
        activation_vectors = np.random.randn(num_subjects, 116).astype(np.float32)
        return images, labels, ages, genders, activation_vectors
    # Load atlas labels once
    atlas_labels = load_nifti_volume(atlas_path).astype(np.int64)
    # Identify metadata file
    meta_file = None
    for fname in ["participants.tsv", "participants.csv", "phenotype.tsv", "phenotype.csv"]:
        candidate = os.path.join(dataset_root, fname)
        if os.path.exists(candidate):
            meta_file = candidate
            break
    if meta_file is None:
        raise FileNotFoundError(f"No metadata file found in {dataset_root}")
    metadata = load_metadata_csv(meta_file)
    images: List[np.ndarray] = []
    labels: List[int] = []
    ages: List[float] = []
    genders: List[int] = []
    activation_vectors: List[List[float]] = []
    for row in metadata:
        # Determine subject identifier
        sid = row.get("subject_id") or row.get("participant_id") or row.get("SUB_ID")
        if sid is None:
            continue
        # Construct path to BOLD file (assumed to reside in sub_<id>/bold.nii*)
        subj_dir = os.path.join(dataset_root, sid)
        # Try a few possible filenames
        bold_path = None
        for base in ["bold.nii.gz", "bold.nii", f"{sid}_bold.nii.gz", f"{sid}_bold.nii"]:
            candidate = os.path.join(subj_dir, base)
            if os.path.exists(candidate):
                bold_path = candidate
                break
        if bold_path is None:
            # Skip subjects without BOLD volume
            continue
        try:
            bold_ts = load_nifti_volume(bold_path)
        except Exception:
            continue
        # Compute 2‑D representation from 4‑D BOLD volume
        img = bold_to_2d_representation(bold_ts, atlas_labels)
        images.append(img)
        # Extract label, age and gender; attempt to convert strings
        lbl = row.get("label") or row.get("diagnosis") or row.get("DX")
        try:
            lbl_int = int(lbl)
        except Exception:
            # Map textual labels to integers; assume binary classification
            lbl_int = 1 if str(lbl).lower() in ("1", "adhd", "autism", "asd", "patient") else 0
        labels.append(lbl_int)
        age_val = row.get("age") or row.get("Age")
        try:
            age_float = float(age_val)
        except Exception:
            age_float = np.nan
        ages.append(age_float)
        gender_val = row.get("gender") or row.get("sex") or row.get("Gender")
        if isinstance(gender_val, str):
            gender_val = gender_val.upper()
            if gender_val in ("M", "MALE", "1"):
                gender_int = 1
            else:
                gender_int = 0
        else:
            try:
                gender_int = int(gender_val)
            except Exception:
                gender_int = 0
        genders.append(gender_int)
        # Compute activation vector by averaging BOLD time courses within each ROI
        vec: List[float] = []
        # For each ROI (1..116) compute mean signal across voxels and across time
        for roi_idx in range(1, 117):
            mask = atlas_labels == roi_idx
            if not np.any(mask):
                vec.append(0.0)
            else:
                ts_roi = bold_ts[mask].mean(axis=0)
                vec.append(float(ts_roi.mean()))
        activation_vectors.append(vec)
    # Convert lists to arrays
    images_arr = np.stack(images).astype(np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    ages_arr = np.array(ages, dtype=np.float32)
    genders_arr = np.array(genders, dtype=np.int64)
    activation_mat = np.array(activation_vectors, dtype=np.float32)
    return images_arr, labels_arr, ages_arr, genders_arr, activation_mat


def main() -> None:
    # Parse command‑line arguments
    args = parse_args()
    # Derive model configuration from size preset and overrides
    model_cfg = get_model_config(args)
    # Determine dataset root and atlas path
    default_roots = {
        "ADHD200": "/path/to/ADHD200",
        "ABIDE": "/path/to/ABIDE",
    }
    dataset_root = args.dataset_path or default_roots.get(args.dataset_name, None)
    atlas_path = args.atlas_path or "/path/to/atlas/AAL116.nii.gz"
    # Load images, labels and activation vectors
    images, labels, ages, genders, activation_vectors = load_dataset(dataset_root, atlas_path)
    # Build vocabulary and discretise activation vectors
    vocab = build_vocab()
    # Choose thresholds: use provided values or defaults
    tau1 = args.tau1 if args.tau1 is not None else 0.15
    tau2 = args.tau2 if args.tau2 is not None else 0.30
    roi_indices = prepare_roi_indices(activation_vectors, tau1, tau2, vocab)
    # Construct dataset and attach ROI indices
    dataset = FMRIImageDataset(images, labels, ages, genders)
    dataset.roi_indices = roi_indices
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # Initialise model with parsed hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RTGMFFModel(
        image_size=(images.shape[1], images.shape[2]),
        patch_size=args.patch_size,
        roi_vocab_size=len(vocab),
        text_embed_dim=args.text_embed_dim,
        num_classes=int(labels.max() + 1) if labels.size > 0 else 2,
        num_wavelet_levels=model_cfg['num_wavelet_levels'],
        hwm_dim=model_cfg['hwm_dim'],
        align_dim=model_cfg['align_dim'],
        num_heads=model_cfg['num_heads'],
        ffn_dropout=args.dropout,
        vit_depth=model_cfg['vit_depth'],
        extra_mlp=args.use_extra_mlp,
        use_ssm=args.use_ssm,
        use_vit=args.use_vit
    ).to(device)
    # Load pretrained weights if specified (only for SSM backbone)
    if args.pretrained:
        # Attempt to load local PyTorch checkpoint; if that fails and a
        # HuggingFace model name is provided, instruct the user to
        # download via the HuggingFace API.  For Mamba models, see the
        # list of available checkpoints on the state‑spaces GitHub page【995923299886983†L410-L433】.
        try:
            state = torch.load(args.pretrained, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"Loaded pretrained weights from {args.pretrained}")
        except Exception as e:
            print(f"Warning: failed to load pretrained weights: {e}."
                  " For models hosted on Hugging Face (e.g., state-spaces/mamba-130m),"
                  " please use the transformers library to download and convert the weights.")
    # Set up optimiser
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Choose classification loss
    class_loss_fn = get_loss_function(args.loss)
    # Training loop
    num_epochs = args.epochs
    alpha, beta = args.alpha, args.beta
    logger = get_logger('rtgmff', logfile=None)
    for epoch in range(1, num_epochs + 1):
        metrics = train_one_epoch(model, loader, optimiser, device, alpha, beta, class_loss_fn=class_loss_fn)
        log_metrics(epoch, metrics, logger=logger, prefix='Train ')
    # Evaluation after training
    eval_metrics = evaluate(model, loader, device)
    print('Evaluation:', eval_metrics)
    # Optionally save model weights
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")
    # Save the run configuration to a timestamped file inside the project’s used_arguments folder
    # Compute the directory relative to this script so that it is packaged with the code
    try:
        current_dir = os.path.dirname(__file__)
        used_dir = os.path.join(current_dir, 'used_arguments')
        os.makedirs(used_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        out_file = os.path.join(used_dir, f"{args.dataset_name}_run_{timestamp}.txt")
        save_args_to_file(args, out_file)
        print(f"Arguments saved to {out_file}")
    except Exception as e:
        print(f"Warning: failed to save arguments: {e}")


if __name__ == '__main__':
    main()