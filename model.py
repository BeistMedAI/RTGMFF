from typing import Tuple, Dict, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletDecomposition(nn.Module):
    """Multi‑level 2‑D Haar wavelet transform producing token sequences."""
    def __init__(self, in_channels: int, num_levels: int, out_dim: int) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.out_dim = out_dim
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        filters = torch.stack([ll, lh, hl, hh], dim=0)
        self.register_buffer('filters', filters)
        self.proj = nn.Linear(in_channels * (4 * num_levels), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        features = []
        current = x
        for _ in range(self.num_levels):
            weight = self.filters.view(4, 1, 2, 2).repeat(C, 1, 1, 1)
            weight = weight.view(4 * C, 1, 2, 2)
            out = F.conv2d(current, weight, stride=2, groups=C)
            features.append(out)
            ll_sub = out[:, :C, :, :]
            current = ll_sub
        concat = torch.cat(features, dim=1)
        B, C_total, h, w = concat.shape
        seq = concat.view(B, C_total, h * w).transpose(1, 2)
        tokens = self.proj(seq)
        return tokens


class SSMLayer(nn.Module):
    """Selective state‑space layer based on the Mamba block.

    Each layer projects the input into a higher dimensional space, applies a depth‑wise
    convolution, computes input‑dependent state space parameters (∆,
    B and C), performs a selective scan over the sequence and then
    projects back to the original dimensionality.  Residual
    connections and gating are applied implicitly within the parent
    encoder.

    Parameters
    ----------
    dim : int
        Feature dimension of the input sequence (``d_model`` in
        the Mamba paper).
    d_state : int, optional
        Dimensionality of the latent state space (``N`` in the paper).
    expand : int, optional
        Expansion factor for the hidden dimension (``E`` in the paper).
    d_conv : int, optional
        Kernel size of the depth‑wise convolution.  A larger kernel
        increases the receptive field.
    dt_rank : int or str, optional
        Rank of the projected step size ∆.  When set to ``'auto'``
        (default), it is computed as ``ceil(dim/16)``【563167105521383†L48-L53】.
    conv_bias : bool, optional
        Whether to include a bias term in the convolution layer.
    bias : bool, optional
        Whether to include biases in the linear projections.
    """

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2,
                 d_conv: int = 4, dt_rank: int | str = 'auto',
                 conv_bias: bool = True, bias: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.d_inner = dim * expand
        # Determine the rank for ∆
        if dt_rank == 'auto':
            self.dt_rank = math.ceil(dim / 16)
        else:
            self.dt_rank = dt_rank
        # Projections
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=bias)
        # Depth‑wise convolution; use groups equal to channels for depth‑wise
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=conv_bias,
            padding=d_conv - 1,
        )
        # Linear mapping to obtain input‑dependent ∆, B and C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        # Project ∆ from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # A is input independent: use log parametrisation as in Mamba【563167105521383†L214-L221】
        # Repeat [1..d_state] across d_inner channels
        A = torch.arange(1, self.d_state + 1).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A.float()))
        # D is also input independent
        self.D = nn.Parameter(torch.ones(self.d_inner))
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the selective state‑space layer.

        Args
        ----
        x : torch.Tensor
            Input sequence of shape ``(B, L, dim)``.

        Returns
        -------
        torch.Tensor
            Output sequence of the same shape as ``x``.
        """
        b, l, d = x.shape
        # Project input and residual; shape (B, L, 2 * d_inner)
        x_and_res = self.in_proj(x)
        x_proj, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        # Depth‑wise convolution operates on (B, d_inner, L)
        h = x_proj.transpose(1, 2)  # (B, d_inner, L)
        h = self.conv1d(h)[:, :, :l]  # Trim padding to maintain length【563167105521383†L245-L248】
        h = h.transpose(1, 2)  # (B, L, d_inner)
        h = F.silu(h)
        # Run the selective state space model
        y = self.ssm(h)
        # Element‑wise modulation by res (gating) as in Mamba【563167105521383†L249-L255】
        y = y * F.silu(res)
        # Project back to original dimension
        out = self.out_proj(y)
        return out

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute selective state space dynamics for a sequence.

        Args
        ----
        x : torch.Tensor
            Sequence of shape ``(B, L, d_inner)``.

        Returns
        -------
        torch.Tensor
            Output sequence of shape ``(B, L, d_inner)``.
        """
        # Compute A and D
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        D = self.D
        # Compute input‑dependent ∆, B and C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Project ∆ to d_inner and ensure positivity
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        # Run selective scan
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u: torch.Tensor, delta: torch.Tensor,
                        A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                        D: torch.Tensor) -> torch.Tensor:
        """Perform the selective scan algorithm (sequential SSM).

        This method implements the discrete state space equations
        described in the Mamba paper.  
        It discretises continuous parameters and applies a sequential
        recurrence to compute the hidden state and output at each
        timestep.  Note that this implementation is sequential and
        does not leverage the parallel scans used in the official
        repository; nonetheless it captures the core dynamics.

        Args
        ----
        u : torch.Tensor
            Input sequence of shape ``(B, L, d_inner)``.
        delta : torch.Tensor
            Step sizes ∆ of shape ``(B, L, d_inner)``.
        A : torch.Tensor
            Continuous state transition matrix of shape ``(d_inner, d_state)``.
        B : torch.Tensor
            Input effect matrix of shape ``(B, L, d_state)``.
        C : torch.Tensor
            Output projection matrix of shape ``(B, L, d_state)``.
        D : torch.Tensor
            Residual scaling vector of shape ``(d_inner,)``.

        Returns
        -------
        torch.Tensor
            Output sequence ``y`` of shape ``(B, L, d_inner)``.
        """
        b, l, d_in = u.shape
        n = A.shape[1]
        # Discretise A and B as in the Mamba minimal implementation【563167105521383†L334-L345】
        # Compute deltaA: (B, L, d_in, n)
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        # Compute deltaB_u: (B, L, d_in, n)
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
        # Sequential scan over time dimension
        x_state = u.new_zeros((b, d_in, n))
        outputs = []
        for i in range(l):
            x_state = deltaA[:, i] * x_state + deltaB_u[:, i]
            # y_t = (x_state @ C_t^T).  C has shape (B, L, n)
            # compute per batch: b,d_in,n  * b,n -> b,d_in
            c_t = C[:, i, :]  # (B, n)
            y_t = torch.einsum('bdn,bn->bd', x_state, c_t)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)  # (B, L, d_in)
        # Add skip term u * D
        y = y + u * D
        return y


class SSMEncoder(nn.Module):
    """Stack of selective state‑space layers for sequence summarisation.

    While the true Mamba model uses diagonal state space kernels and hardware‑aware parallelism, our
    simplified encoder captures the essential property of selectively
    propagating or forgetting information along the sequence.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input sequence features.
    depth : int
        Number of SSM layers to stack.
    """

    def __init__(self, embed_dim: int, depth: int = 4) -> None:
        super().__init__()
        # Stack multiple selective state‑space layers (Mamba blocks)
        self.layers = nn.ModuleList([
            SSMLayer(embed_dim) for _ in range(depth)
        ])

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Process the sequence through stacked SSM layers and pool.

        Args
        ----
        seq : torch.Tensor
            Input sequence of shape ``(B, L, d)``.

        Returns
        -------
        torch.Tensor
            A fixed‑length representation obtained by averaging the
            processed sequence.
        """
        out = seq
        for layer in self.layers:
            out = layer(out)
        # Mean pooling over the sequence dimension
        return out.mean(dim=1)


class VisionTransformerEncoder(nn.Module):
    """A minimalist Vision Transformer encoder for image representations.

    Inspired by the ViT architecture, this module divides an image
    into fixed‑size patches, flattens them and projects each patch to
    a high‑dimensional vector.  A learnable positional embedding is
    added to preserve spatial order.  The resulting sequence is
    processed by a stack of Transformer encoder layers, each
    comprising multi‑head self‑attention and feed‑forward sub‑layers
    with residual connections.  
    The final representation is obtained by average pooling over the patch
    tokens.  Positional embeddings and the CLS token common in
    classification ViTs are omitted for simplicity, as the RTGMFF
    pipeline already applies a separate alignment module.

    Parameters
    ----------
    image_size : Tuple[int, int]
        Height and width of the input images.
    patch_size : int
        Spatial size of each square patch.
    in_channels : int
        Number of input channels (e.g. 3 for ALFF/fALFF/ReHo maps).
    embed_dim : int
        Dimensionality of the patch embeddings.
    num_heads : int
        Number of attention heads in each Transformer block.
    depth : int
        Number of Transformer encoder layers.
    """

    def __init__(self, image_size: Tuple[int, int], patch_size: int,
                 in_channels: int, embed_dim: int, num_heads: int, depth: int = 4) -> None:
        super().__init__()
        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0, "image size must be divisible by patch size"
        self.patch_size = patch_size
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.embed_dim = embed_dim
        # Linear projection of flattened patches
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                                     stride=patch_size)
        # Positional embedding for each patch token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(depth)
        ])
        # Layer normalisation at the end
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        patches = self.patch_embed(x)  # (B, E, H_p, W_p)
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, E)
        tokens = patches + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        # Average pool over all patch tokens to obtain a global feature
        return tokens.mean(dim=1)


class SelectiveScan(nn.Module):
    """Soft token selection via learned importance weights."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.score(x).squeeze(-1)
        attn = F.softmax(weights, dim=-1).unsqueeze(-1)
        summary = (x * attn).sum(dim=1)
        return summary


class ConvFFN(nn.Module):
    """Two‑layer feed‑forward network with GELU activation."""
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        layers: List[nn.Module] = [
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossScaleTransformerEncoder(nn.Module):
    """Fuse local and global representations via cross attention."""
    def __init__(self, image_size: Tuple[int, int], patch_size: int,
                 in_channels: int, embed_dim: int, num_heads: int,
                 vit_depth: int = 4) -> None:
        super().__init__()
        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0, "image size must be divisible by patch size"
        self.patch_size = patch_size
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                                     stride=patch_size)
        self.downsample = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.cross_att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.vit_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(vit_depth)
        ])
    def forward(self, x: torch.Tensor, local_features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Patch embedding and flattening
        patches = self.patch_embed(x)
        H_p, W_p = patches.shape[2], patches.shape[3]
        patches = patches.flatten(2).transpose(1, 2)
        # Downsample for queries
        down = self.downsample(self.patch_embed(x))
        down = down.flatten(2).transpose(1, 2)
        # Project local features into key and value spaces
        proj_k = nn.Linear(local_features.size(-1), self.embed_dim).to(local_features.device)
        proj_v = nn.Linear(local_features.size(-1), self.embed_dim).to(local_features.device)
        K = proj_k(local_features)
        V = proj_v(local_features)
        # Cross attention
        attn_output, _ = self.cross_att(down, K, V)
        # Upsample attention output to match patch sequence
        attn_output = attn_output.unsqueeze(2).permute(0, 2, 1, 3).contiguous()
        h_t, w_t = H_p, W_p
        attn_upsampled = F.interpolate(attn_output.reshape(B, 1, h_t, w_t, self.embed_dim),
                                       size=(h_t, w_t), mode='nearest')
        attn_upsampled = attn_upsampled.view(B, h_t * w_t, self.embed_dim)
        fused = patches + attn_upsampled
        for block in self.vit_blocks:
            fused = block(fused)
        global_feat = fused.mean(dim=1)
        return global_feat


class ASAM(nn.Module):
    """Adaptive semantic alignment module for visual–text fusion."""
    def __init__(self, visual_dim: int, text_dim: int, align_dim: int, num_classes: int) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, align_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, align_dim, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(1, align_dim),
            nn.ReLU(),
            nn.Linear(align_dim, num_classes)
        )
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_proj = self.visual_proj(z)
        t_proj = self.text_proj(t)
        cos_sim = F.cosine_similarity(z_proj, t_proj, dim=-1, eps=1e-8).unsqueeze(-1)
        logits = self.classifier(cos_sim)
        return logits, z_proj, t_proj


class RTGMFFModel(nn.Module):
    """Composite model implementing the RTGMFF pipeline with optional backbones.

    The RTGMFF model fuses hierarchical wavelet tokens derived from
    2‑D fMRI slices with discretised ROI activation sequences and
    demographic variables.  It offers a flexible choice of
    sequence‑processing backbones: a cross‑scale Transformer encoder
    (default), a simple state space encoder inspired by the Mamba
    architecture.  

    Parameters
    ----------
    image_size : Tuple[int, int]
        Height and width of the input 2‑D maps.
    patch_size : int
        Spatial size of each square patch for Transformer/ViT.
    roi_vocab_size : int
        Size of the ROI token vocabulary.
    text_embed_dim : int
        Dimensionality of the text (ROI) embeddings.
    num_classes : int
        Number of diagnosis classes.
    num_wavelet_levels : int, optional
        Levels of the Haar wavelet decomposition.  Defaults to 2.
    hwm_dim : int, optional
        Output dimension of the wavelet token projector.  Defaults to 128.
    align_dim : int, optional
        Dimension of the alignment projection used in ASAM.  Defaults to 64.
    num_heads : int, optional
        Number of attention heads in the Transformer/ViT encoder.  Defaults to 4.
    ffn_dropout : float, optional
        Dropout rate in the feed‑forward networks of the HWM branch.
    vit_depth : int, optional
        Number of Transformer blocks in the cross‑scale or ViT encoder.  Defaults to 4.
    extra_mlp : bool, optional
        Whether to apply an additional MLP to the fused visual features.
    use_ssm : bool, optional
        If ``True``, replace the cross‑scale Transformer with a state
        space encoder.  ``use_vit`` must be ``False`` when this is enabled.
    use_vit : bool, optional
        If ``True``, replace the cross‑scale Transformer with a pure
        Vision Transformer encoder.  ``use_ssm`` must be ``False`` when
        this is enabled.
    """

    def __init__(self, image_size: Tuple[int, int], patch_size: int,
                 roi_vocab_size: int, text_embed_dim: int, num_classes: int,
                 num_wavelet_levels: int = 2, hwm_dim: int = 128,
                 align_dim: int = 64, num_heads: int = 4,
                 ffn_dropout: float = 0.0,
                 vit_depth: int = 4,
                 extra_mlp: bool = False,
                 use_ssm: bool = False,
                 use_vit: bool = False) -> None:
        super().__init__()
        assert not (use_ssm and use_vit), "Only one of use_ssm or use_vit may be true"
        H, W = image_size
        # Token embedding for discretised ROI activations
        self.text_embedding = nn.Embedding(roi_vocab_size, text_embed_dim)
        # Haar wavelet module producing multi‑scale token sequences
        self.hwm = HaarWaveletDecomposition(in_channels=3, num_levels=num_wavelet_levels,
                                             out_dim=hwm_dim)
        # Split the wavelet token sequence into fixed subsequences for
        # selective scanning and feed‑forward refinement
        self.num_subseq = 4
        self.selective_scans = nn.ModuleList([
            SelectiveScan(hwm_dim) for _ in range(self.num_subseq)
        ])
        self.ffns = nn.ModuleList([
            ConvFFN(hwm_dim, dropout=ffn_dropout) for _ in range(self.num_subseq)
        ])
        self.hwm_output_dim = hwm_dim * self.num_subseq
        # Sequence processing backbone: choose between Transformer, SSM or ViT
        self.use_ssm = use_ssm
        self.use_vit = use_vit
        if use_vit:
            # A pure ViT encoder processes the raw image directly
            self.vit = VisionTransformerEncoder(
                image_size=image_size,
                patch_size=patch_size,
                in_channels=3,
                embed_dim=self.hwm_output_dim,
                num_heads=num_heads,
                depth=vit_depth
            )
        elif use_ssm:
            # Process wavelet tokens with a stack of SSM layers
            self.ssm = SSMEncoder(embed_dim=self.hwm_output_dim, depth=vit_depth)
        else:
            # Default cross‑scale Transformer encoder
            self.cste = CrossScaleTransformerEncoder(
                image_size=image_size,
                patch_size=patch_size,
                in_channels=3,
                embed_dim=self.hwm_output_dim,
                num_heads=num_heads,
                vit_depth=vit_depth
            )
        # Adaptive semantic alignment module for fusing visual and text features
        self.asam = ASAM(
            visual_dim=self.hwm_output_dim,
            text_dim=text_embed_dim,
            align_dim=align_dim,
            num_classes=num_classes
        )
        # FiLM MLP to modulate fused visual features by demographic variables
        self.film = nn.Sequential(
            nn.Linear(3, self.hwm_output_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hwm_output_dim * 2, self.hwm_output_dim * 2)
        )
        # Optional extra MLP to enrich fused features
        self.use_extra_mlp = extra_mlp
        if self.use_extra_mlp:
            self.extra_mlp = nn.Sequential(
                nn.Linear(self.hwm_output_dim, self.hwm_output_dim),
                nn.ReLU(),
                nn.Linear(self.hwm_output_dim, self.hwm_output_dim)
            )

    def forward(self, images: torch.Tensor, roi_indices: torch.Tensor,
                ages: torch.Tensor, genders: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Compute wavelet token sequence
        hwm_tokens = self.hwm(images)  # (B, L, hwm_dim)
        L = hwm_tokens.size(1)
        # Split sequence into equally sized subsequences
        split_size = L // self.num_subseq
        subseqs = hwm_tokens.split(split_size, dim=1)
        refined: List[torch.Tensor] = []
        # Apply selective scanning and feed‑forward refinement to each subsequence
        for i in range(self.num_subseq):
            summary = self.selective_scans[i](subseqs[i])  # (B, hwm_dim)
            refined_feat = self.ffns[i](summary)          # (B, hwm_dim)
            refined.append(refined_feat)
        # Concatenate refined features from all subsequences
        local_feat = torch.cat(refined, dim=-1)  # (B, hwm_output_dim)
        # Compute a global representation using the chosen backbone
        if self.use_vit:
            # ViT takes the raw image as input
            global_feat = self.vit(images)  # (B, hwm_output_dim)
        elif self.use_ssm:
            # SSM processes the sequence of wavelet tokens
            global_feat = self.ssm(hwm_tokens)  # (B, hwm_output_dim)
        else:
            # Default cross‑scale Transformer encoder
            global_feat = self.cste(images, hwm_tokens)  # (B, hwm_output_dim)
        # Fuse local and global features by addition
        fused_visual = local_feat + global_feat
        # Demographic FiLM conditioning: normalise age and one‑hot encode gender
        age_norm = (ages - ages.mean()) / (ages.std() + 1e-8)
        gender_onehot = F.one_hot(genders, num_classes=2).float()
        d = torch.stack([age_norm, gender_onehot[:, 1], gender_onehot[:, 0]], dim=1)
        film_params = self.film(d)
        gamma, beta = film_params.chunk(2, dim=-1)
        fused_visual = gamma * fused_visual + beta
        # Apply optional extra MLP for richer interactions
        if self.use_extra_mlp:
            fused_visual = self.extra_mlp(fused_visual)
        # Embed discretised ROI indices and average over sequence
        text_embeds = self.text_embedding(roi_indices)
        text_feat = text_embeds.mean(dim=1)
        # Align and classify
        logits, z_proj, t_proj = self.asam(fused_visual, text_feat)
        aux = {
            'z_proj': z_proj,
            't_proj': t_proj,
            'fused_visual': fused_visual,
            'text_feat': text_feat,
        }
        return logits, aux
