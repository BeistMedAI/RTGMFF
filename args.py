import argparse
from typing import Dict


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for the RTGMFF pipeline.

    Returns
    -------
    argparse.Namespace
        A namespace containing the parsed arguments and their values.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate the RTGMFF model on fMRI datasets."
    )
    # Dataset selection
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["ADHD200", "ABIDE"],
        default="ADHD200",
        help="Name of the neuroimaging dataset to use."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Root directory containing the fMRI dataset.  When left unset,"
            " a sensible default is assumed for the selected dataset (e.g."
            " /path/to/ADHD200 or /path/to/ABIDE)."
        ),
    )
    parser.add_argument(
        "--atlas-path",
        type=str,
        default=None,
        help="Path to a 3‑D atlas (NIfTI) mapping voxels to one of the 116 ROIs."
    )
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Mini‑batch size for training and evaluation."
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for the AdamW optimiser."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularisation strength) for AdamW."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Coefficient α for the alignment loss term."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.2,
        help="Coefficient β for the Gram matrix regularisation term."
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["ce", "focal", "label_smooth", "ls"],
        default="ce",
        help=(
            "Type of classification loss to use: 'ce' for cross entropy,"
            " 'focal' for focal loss, 'label_smooth' or 'ls' for label smoothing."
        ),
    )
    # Model architecture options
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "base", "large"],
        default="base",
        help="Select the base dimensionality for the model (see model.py for details)."
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=8,
        help="Image patch size for the Transformer/ViT branches."
    )
    parser.add_argument(
        "--wavelet-levels",
        type=int,
        default=None,
        help="Number of decomposition levels in the Haar wavelet module."
    )
    parser.add_argument(
        "--hwm-dim",
        type=int,
        default=None,
        help="Dimensionality of the wavelet token projector."
    )
    parser.add_argument(
        "--align-dim",
        type=int,
        default=None,
        help="Dimensionality of the alignment projection in ASAM."
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Number of attention heads in the Transformer encoder."
    )
    parser.add_argument(
        "--vit-depth",
        type=int,
        default=None,
        help="Number of Transformer encoder blocks in the cross‑scale or ViT encoder."
    )
    parser.add_argument(
        "--text-embed-dim",
        type=int,
        default=64,
        help="Dimensionality of the ROI text embedding."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate for feed‑forward networks inside the HWM branch."
    )
    parser.add_argument(
        "--use-ssm",
        action="store_true",
        help=(
            "Replace the cross‑scale Transformer encoder with a Mamba‑inspired "
            "state‑space model (SSM).  This backbone selectively propagates or "
            "forgets information along token sequences【859331777488541†L50-L65】."
        ),
    )
    parser.add_argument(
        "--use-vit",
        action="store_true",
        help=(
            "Use a pure Vision Transformer (ViT) backbone for the image branch"
            " instead of the default cross‑scale encoder.  ViTs split an image"
            " into fixed‑size patches, embed each patch and process the sequence"
            " with multi‑head self‑attention【141199850673451†L352-L362】."
        ),
    )
    parser.add_argument(
        "--use-extra-mlp",
        action="store_true",
        help="Enable an additional MLP on the fused visual representation."
    )
    parser.add_argument(
        "--tau1",
        type=float,
        default=None,
        help="Lower threshold τ₁ for discretising activation vectors."
    )
    parser.add_argument(
        "--tau2",
        type=float,
        default=None,
        help="Upper threshold τ₂ for discretising activation vectors."
    )
    # Paths for saving and loading
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help=(
            "Optional path or HuggingFace model name from which to load"
            " pretrained weights for the state‑space backbone.  For example,"
            " state‑spaces/mamba-130m or a local .pth file【995923299886983†L410-L433】."
        ),
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="File path to save the trained model weights (PyTorch .pth format)."
    )
    args = parser.parse_args()
    return args


def get_model_config(args: argparse.Namespace) -> Dict[str, int]:
    """Derive model hyperparameters from the selected model size.

    The RTGMFF model uses a handful of architectural parameters that
    scale together.  Rather than forcing the user to set every value
    individually, we provide sensible defaults for each size tier.  The
    values can still be overridden explicitly by supplying the
    corresponding command‑line flags (e.g. ``--hwm-dim``).

    Parameters
    ----------
    args : argparse.Namespace
        The parsed argument namespace.

    Returns
    -------
    Dict[str, int]
        A dictionary containing ``num_wavelet_levels``, ``hwm_dim``,
        ``align_dim``, ``num_heads`` and ``vit_depth``.
    """
    presets = {
        "small": {"num_wavelet_levels": 1, "hwm_dim": 32,  "align_dim": 16, "num_heads": 2, "vit_depth": 2},
        "base":  {"num_wavelet_levels": 2, "hwm_dim": 64,  "align_dim": 32, "num_heads": 4, "vit_depth": 4},
        "large": {"num_wavelet_levels": 3, "hwm_dim": 128, "align_dim": 64, "num_heads": 8, "vit_depth": 6},
    }
    cfg = presets.get(args.model_size, presets["base"]).copy()
    if args.wavelet_levels is not None:
        cfg["num_wavelet_levels"] = args.wavelet_levels
    if args.hwm_dim is not None:
        cfg["hwm_dim"] = args.hwm_dim
    if args.align_dim is not None:
        cfg["align_dim"] = args.align_dim
    if args.num_heads is not None:
        cfg["num_heads"] = args.num_heads
    if args.vit_depth is not None:
        cfg["vit_depth"] = args.vit_depth
    return cfg


def save_args_to_file(args: argparse.Namespace, filepath: str) -> None:
    """Serialise the provided namespace into a human‑readable text file.

    The output file contains one flag per line in the order they appear in
    the namespace.  This makes it easy to reconstruct the exact command
    used for a given experiment.  Only attributes that differ from
    ``None`` are written out.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments to save.
    filepath : str
        Destination path for the text file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for key, value in vars(args).items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    f.write(f"--{key}\n")
            else:
                f.write(f"--{key} {value}\n")