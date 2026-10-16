# Adapted from masked_reference commit 3ff606831e48be8069cb42a2afffa666051d53fa; see CODEC_NOTES.md.
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import math
from .compress_sequence import CausalCompressSelfAttention, DownsampleBy2
from .decompress_sequence import UpSampleBy2
from .attention_utils import Rotary, rms_norm, apply_rotary_emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = torch.einsum("...i,i->...i",t.unsqueeze(-1).float(),freqs) # (b,s,f)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # (b,s,2f)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embeddings (RoPE) and QK-norm.

    Uses PyTorch's ``scaled_dot_product_attention`` (FlashAttention-2 when available).
    Key design choices:
      - **RoPE**: rotary embeddings encode relative positions without adding extra params.
      - **QK-norm**: RMS-normalising Q and K stabilises training (suggested by @Grad62304977).
      - **Zero-init projection**: the output projection starts at zero so the residual stream
        is initially unchanged; this eases early training.
      - **Bidirectional** mask (all-ones): the encoder does not need causal masking.

    Args:
        config: GPTConfig with ``n_embd`` and ``n_head``.
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
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = rms_norm(q, (q.size(-1),)), rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # Bidirectional attention: all positions attend to all others (full mask of ones).
        full_mask = torch.ones((T, T), dtype=q.dtype, device=q.device)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=full_mask,
        )
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """Position-wise feed-forward network with squared-ReLU activation.

    Architecture: Linear(n_embd → 4*n_embd) → relu²  → Linear(4*n_embd → out_dim)

    Why ``relu(x).square()`` instead of GELU?  It is ~1-2% better on language
    modelling benchmarks (https://arxiv.org/abs/2109.08668v2) while being
    simpler to compute.

    Args:
        config:       GPTConfig with ``n_embd`` and ``output_dim``.
        is_last_layer: If True, project to ``config.output_dim`` instead of
                       ``config.n_embd`` (used only for the final output head).
    """

    def __init__(self, config, is_last_layer=False):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        out_dim = config.n_embd if not is_last_layer else config.output_dim
        self.c_proj  = nn.Linear(4 * config.n_embd, out_dim, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # relu² — ~1-2% better than GELU (arxiv 2109.08668)
        x = self.c_proj(x)
        return x

class LocalConvModule(nn.Module):
    """Conformer-style local convolutional module with GLU gating.

    Captures short-range (local) structure that global attention may overlook.
    Pipeline:
      1. Pointwise conv expands channels to 2*hidden_dim.
      2. GLU gating: output = act_half * sigmoid(gate_half).
      3. Depthwise conv (kernel_size) mixes nearby positions.
      4. SiLU activation + pointwise conv restore channel count.

    This module is applied *in addition to* self-attention inside each ``Block``,
    following the Conformer architecture (Gulati et al., 2020).

    Args:
        hidden_dim:  Number of channels.
        kernel_size: Receptive field of the depthwise conv.
        dropout:     Dropout probability applied after the final projection.
    """

    def __init__(self, hidden_dim, kernel_size=7, dropout=0.1):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=hidden_dim)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        b, t, c = x.shape
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.pointwise_conv1(x)
        # GLU gating along channel dim
        x_gate, x_act = x.chunk(2, dim=1)
        x = x_act * torch.sigmoid(x_gate)
        x = self.depthwise_conv(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, C)
        return x

class Block(nn.Module):
    """Single transformer block: Attention + LocalConv + MLP with Adaptive LayerNorm (AdaLN).

    Residual structure (all norms are zero-affine, conditioning comes from AdaLN):
      x = x_res + gate_msa * Attention(AdaLN(x))
      x = x     + LocalConvModule(AdaLN(x))
      x = x     + gate_mlp * MLP(AdaLN(x))

    AdaLN modulation: a conditioning vector ``c`` (e.g. timestep embedding) predicts
    6 scalars — (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) —
    that modulate each sub-layer.  This lets the model behave differently at different
    noise levels without changing any weights.

    Args:
        config:      GPTConfig.
        is_compress: If True, use ``CausalCompressSelfAttention`` (halves sequence length)
                     and ``DownsampleBy2`` on the residual path.  Used for exactly one
                     middle block in the encoder.
    """

    def __init__(self, config, is_compress=False):
        super().__init__()
        self.attn = CausalSelfAttention(config) if not is_compress else CausalCompressSelfAttention(config)
        self.mlp = MLP(config)
        self.norm1 = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.norm_conv = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 6 * config.n_embd, bias=True)
        )
        self.conv_module = LocalConvModule(config.n_embd, kernel_size=7, dropout=config.conv_dropout)
        self.res_down = DownsampleBy2(config.n_embd) if is_compress else nn.Identity()
        self.is_compress = is_compress

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        res = self.res_down(x) if self.is_compress else x
        x = res + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # Conformer-style local convolutional module
        x = x + self.conv_module(modulate(self.norm_conv(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
    output_dim : int = 1024
    conv_dropout : float = 0.1

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, config):
        super().__init__()
        self.norm_final = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(config.n_embd, config.output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 2 * config.n_embd, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TokenProjection(nn.Linear):
    """Linear one-hot projection via lookup; weight/bias checkpoint layout is unchanged."""
    def forward(self, inputs):
        if inputs.dtype == torch.long:
            result = F.embedding(inputs, self.weight.t())
            # Embedding is not an autocast operator, while the original Linear
            # is. Match its output precision to preserve mixed-precision paths.
            autocast = torch.is_autocast_enabled(inputs.device.type) and self.weight.dtype != torch.float64
            if autocast:
                result = result.to(torch.get_autocast_dtype(inputs.device.type))
            if self.bias is not None:
                bias = self.bias.to(result.dtype) if autocast else self.bias
                result = result + bias
            return result
        return super().forward(inputs)


class GPT(nn.Module):
    """GPT-style transformer used as encoder or decoder inside the VQ-VAE.

    The same architecture serves three roles depending on flags:
      - **Plain encoder/decoder** (``is_compress=False``, ``is_decompress=False``):
        A standard stack of ``Block``s with a linear input projection and a
        ``FinalLayer`` output head.
      - **Compressing encoder** (``is_compress=True``):
        The middle block uses ``CausalCompressSelfAttention``, so the sequence
        length is halved at that layer (e.g. 64 tokens → 32).
      - **Decompressing decoder** (``is_decompress=True``):
        Before the first block, ``UpSampleBy2`` doubles the sequence length back
        (32 → 64), optionally fusing a skip connection from the encoder.

    Args:
        config:        GPTConfig (vocab_size, n_layer, n_head, n_embd, output_dim).
        is_compress:   Build a compressing encoder (halves sequence at mid-layer).
        is_decompress: Build a decompressing decoder (doubles sequence at start).
        need_fuse:     When ``is_decompress=True``, whether to use a gated skip
                       connection from the corresponding encoder layer.

    Forward args:
        t:                     Timestep tensor (B,) or (B, T) for conditioning.
        xt:                    Input one-hot or soft distribution (B, T, vocab_size).
        targets:               Token indices for cross-entropy loss (optional).
        return_logits:         Whether to return logits (set False to save memory).
        x_from_previous_stage: Skip-connection from encoder (required when is_decompress=True).

    Returns:
        logits: (B, T, output_dim) or None if ``return_logits=False``.
    """

    def __init__(self, config, is_compress=False, is_decompress=False, need_fuse=False):
        super().__init__()
        self.config = config
        self.is_compress = is_compress
        self.is_decompress = is_decompress
        self.need_fuse = need_fuse
        assert config.n_layer % 2 == 1
        # The middle block is the downsampling block when is_compress=True.
        mid = config.n_layer // 2
        self.transformer = nn.ModuleDict(dict(
            wte = TokenProjection(config.vocab_size, config.n_embd, bias=False),
            h = nn.ModuleList([
                Block(config, is_compress=(i == mid and is_compress))
                for i in range(config.n_layer)
            ]),
        ))
        if is_decompress:
            self.decompress = UpSampleBy2(config.n_embd, need_fuse)
        self.t_embedder = TimestepEmbedder(config.n_embd)
        self.final_layer = FinalLayer(config)

    def forward_hidden(self, t, xt, x_from_previous_stage=None):
        """Return hidden states before the vocabulary head (for chunked CE)."""
        c = self.t_embedder(t)
        x = self.transformer.wte(xt)
        for i, block in enumerate(self.transformer.h):
            if self.is_decompress and i == 0:
                x = self.decompress(x, x_from_previous_stage)
            x = block(x, c)
        return x, c

    def forward(self, t, xt, targets=None, return_logits=True, x_from_previous_stage=None):
        x, c = self.forward_hidden(t, xt, x_from_previous_stage)
        logits = self.final_layer(x, c).float()
        # Historical callers did not consume the internally computed CE.
        if targets is not None:
            F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits if return_logits else None
