# Adapted from masked_reference commit 3ff606831e48be8069cb42a2afffa666051d53fa; see CODEC_NOTES.md.
import torch
import torch.nn.functional as F

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached or self.cos_cached.device != x.device:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

def rms_norm(x, normalized_shape, eps=1e-6):
    """
    Custom RMS normalization that falls back to F.rms_norm if available.

    Args:
        x: Input tensor
        normalized_shape: Shape of the normalization window
        eps: Small constant for numerical stability
    """
    # Try using F.rms_norm if available (PyTorch >= 2.0)
    if hasattr(F, 'rms_norm'):
        return F.rms_norm(x, normalized_shape, eps=eps)
    else:
        # Fallback implementation for older PyTorch versions
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
