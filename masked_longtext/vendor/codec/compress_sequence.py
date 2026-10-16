# Adapted from masked_reference commit 3ff606831e48be8069cb42a2afffa666051d53fa; see CODEC_NOTES.md.


import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_utils import Rotary, rms_norm, apply_rotary_emb


class AntiAliasDownsample1D(nn.Module):
    """Apply a low-pass Gaussian filter and then downsample by 2.

    Before halving the sequence length we blur the signal with a [1, 2, 1]/4
    kernel to remove high-frequency components that would otherwise alias.
    This is the standard anti-aliasing trick used in signal processing.

    Args:
        d: Number of channels (embedding dimension).

    Input:  (B, N, d)  — batch, sequence length, channels
    Output: (B, N//2, d)
    """

    def __init__(self, d):
        super().__init__()
        kernel = torch.tensor([1., 2., 1.]) / 4.
        # Shape (d, 1, 3): one filter per channel, run as depthwise conv.
        self.register_buffer('kernel', kernel.view(1, 1, 3).repeat(d, 1, 1))
        self.d = d

    def forward(self, x):
        B, N, d = x.shape
        x_ = x.transpose(1, 2)                    # (B, d, N) for Conv1d
        y = F.conv1d(x_, self.kernel, stride=2, padding=1, groups=d)
        return y.transpose(1, 2)                   # (B, N//2, d)


class PairDownsample(nn.Module):
    """Merge adjacent token pairs and project back to the original dimension.

    Each pair of consecutive tokens is concatenated (giving a 2*d vector) and
    then projected back to d via a small MLP.  This lets the model *learn* how
    to combine neighboring tokens rather than hard-coding a fixed pooling rule.

    Args:
        d:           Channel dimension.
        hidden_mult: Width multiplier for the internal MLP layer.

    Input:  (B, N, d)  — N must be even
    Output: (B, N//2, d)
    """

    def __init__(self, d, hidden_mult=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, hidden_mult * 2 * d),
            nn.GELU(),
            nn.Linear(hidden_mult * 2 * d, d),
        )

    def forward(self, x):
        B, N, d = x.shape
        assert N % 2 == 0, "Sequence length must be even for PairDownsample."
        # View pairs of tokens as a single 2*d vector, then project.
        x = x.view(B, N // 2, 2, d).reshape(B, N // 2, 2 * d)
        return self.proj(x)


class DownsampleBy2(nn.Module):
    """Halve the sequence length using a blend of anti-aliasing and learned pair merging.

    Combines two complementary strategies:
      1. ``AntiAliasDownsample1D`` — signal-processing low-pass filter.
      2. ``PairDownsample``        — learned merging of adjacent tokens.

    Their outputs are averaged (weight 0.5 each) so neither dominates.

    Args:
        d: Channel dimension.

    Input:  (B, N, d)
    Output: (B, N//2, d)
    """

    def __init__(self, d):
        super().__init__()
        self.lp = AntiAliasDownsample1D(d)
        self.mix = PairDownsample(d)

    def forward(self, x):
        x_lp = self.lp(x)        # low-pass filtered & strided
        x_pairs = self.mix(x)    # learned pair merge
        return 0.5 * (x_lp + x_pairs)


class CausalCompressSelfAttention(nn.Module):
    """Self-attention where queries are compressed by 2× before attention.

    Standard self-attention produces an output with the same length as the
    input.  Here the query sequence is first compressed with ``DownsampleBy2``
    so the attention outputs half as many tokens.  Keys and values still come
    from the full-resolution input, which gives every compressed query access
    to all original tokens.

    The attention mask is set to all-ones (no causal masking) because the
    encoder does not need to be causal.

    Args:
        config: GPTConfig with fields ``n_head``, ``n_embd``.

    Input:  (B, T, n_embd)
    Output: (B, T//2, n_embd)

    TODO: consider a smarter compression strategy later.
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.compress_q = DownsampleBy2(self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        # Compress queries to T//2; keys & values keep full resolution T.
        x_for_query = self.compress_q(x)
        q = self.c_q(x_for_query).view(B, T // 2, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        cos_q, sin_q = self.rotary(q)   # [1, T//2, 1, head_dim/2]
        cos_k, sin_k = self.rotary(k)   # [1, T,    1, head_dim/2]
        q, k = rms_norm(q, (q.size(-1),)), rms_norm(k, (k.size(-1),))

        q = apply_rotary_emb(q, cos_q, sin_q)
        k = apply_rotary_emb(k, cos_k, sin_k)

        # All-ones mask: shape [query_len, key_len] = [T//2, T].
        # After compression, there are T//2 query positions (one per output token)
        # and T key positions (all original tokens). Every query can attend to every key.
        attn_mask = torch.ones((T // 2, T), dtype=q.dtype, device=q.device)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attn_mask,
        )
        y = y.transpose(1, 2).contiguous().view(B, T // 2, C)
        y = self.c_proj(y)
        return y
