# Adapted from masked_reference commit 3ff606831e48be8069cb42a2afffa666051d53fa; see CODEC_NOTES.md.


import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """Blend two same-shaped tensors with a learned per-position gate.

    Given an upsampled tensor ``x_up`` and a skip-connection tensor ``x_skip``
    (both shape ``(B, N, dim)``), a small network predicts a gate ``α ∈ (0,1)``
    at each position.  The output is:

        fused = α * x_skip + (1 - α) * x_up

    When ``α = 1`` the skip connection passes through unchanged; when ``α = 0``
    only the upsampled signal survives.  The model learns the right balance.

    Args:
        dim: Channel dimension of both inputs.
    """

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x_up, x_skip):
        gate_inp = torch.cat([x_up, x_skip], dim=-1)   # (B, N, 2*dim)
        alpha = self.gate(gate_inp)                     # (B, N, dim), values in (0, 1)
        fused = alpha * x_skip + (1 - alpha) * x_up
        return fused


class UpSampleBy2(nn.Module):
    """Double the sequence length using learnable convolutions with an optional skip connection.

    Upsampling pipeline:
      1. Nearest-neighbor 2× upsample (cheap, no learnable params).
      2. Depthwise conv (kernel=5) to smooth the repeated values.
      3. Pointwise conv to mix channels.
      4. SiLU activation.

    When ``need_fuse=True`` a second branch processes the skip-connection tensor
    ``x_skip`` through the same upsample+conv pipeline, and then a
    ``GatedFusion`` module blends it with the latent before the main upsample.
    This lets the decoder use fine-grained information from the encoder path
    (U-Net style skip connections).

    Args:
        d:         Channel dimension.
        need_fuse: Whether to accept and fuse a skip-connection tensor.
                   Set to True for intermediate decoder layers, False for the
                   last layer where no skip tensor is available.

    TODO: consider a smarter upsampling strategy later.
    """

    def __init__(self, d, need_fuse=True):
        super().__init__()
        # Main upsample branch: nearest x2 → depthwise → pointwise → SiLU
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.dwconv = nn.Conv1d(d, d, kernel_size=5, padding=2, groups=d)
        self.pwconv = nn.Conv1d(d, d, kernel_size=1)
        self.act = nn.SiLU()
        if need_fuse:
            # Skip-connection branch: same pipeline as the main branch.
            self.fuse_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.fuse_dwconv = nn.Conv1d(d, d, kernel_size=5, padding=2, groups=d)
            self.fuse_pwconv = nn.Conv1d(d, d, kernel_size=1)
            self.fuse_act = nn.SiLU()
            self.fuse = GatedFusion(d)
        else:
            self.fuse = None
        self.need_fuse = need_fuse

    def forward(self, x_latent, x_skip):
        """
        Args:
            x_latent: Compressed latent tensor, shape (B, N, d).
            x_skip:   Skip-connection tensor from the encoder, shape (B, 2*N, d).
                      Ignored when ``need_fuse=False``.

        Returns:
            Upsampled tensor of shape (B, 2*N, d).
        """
        if self.need_fuse:
            # Process skip connection: (B, 2*N, d) → (B, d, 2*N) → conv → (B, 2*N, d)
            x_skip = x_skip.transpose(1, 2)
            x_skip = self.fuse_upsample(x_skip)
            x_skip = self.fuse_dwconv(x_skip)
            x_skip = self.fuse_act(x_skip)
            x_skip = self.fuse_pwconv(x_skip)
            x_skip = x_skip.transpose(1, 2)
            # Blend latent and skip using the gated fusion.
            x_latent = self.fuse(x_latent, x_skip)
        # Main upsample: (B, N, d) → (B, d, N) → upsample → conv → (B, 2*N, d)
        x_latent = x_latent.transpose(1, 2)
        x_up = self.upsample(x_latent)
        x_up = self.dwconv(x_up)
        x_up = self.act(x_up)
        x_up = self.pwconv(x_up)
        out = x_up.transpose(1, 2)
        return out
