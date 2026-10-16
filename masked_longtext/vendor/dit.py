"""Pure PyTorch port of the selected reference MDLM DiT.

Source: masked_reference commit 3ff606831e48be8069cb42a2afffa666051d53fa,
latent_diffusion/mdlm/models2/dit.py. Architecture and initialization preserved;
einops/FlashAttention/Hub/OmegaConf and global JIT switches removed. See CORE_NOTES.md.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x):
        # Accumulate normalization in fp32, including mixed precision training.
        out = F.layer_norm(x.float(), (self.dim,)).to(x.dtype)
        return out * self.weight.to(x.dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, cond_dim, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, cond_dim),
                                 nn.SiLU(), nn.Linear(cond_dim, cond_dim))

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(-math.log(10000) *
                                torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float()[:, None] * frequencies[None]
        emb = torch.cat((args.cos(), args.sin()), dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class Rotary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("inv_freq", 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x):
        # No device/dtype-stale Python cache; position 0 is the beginning of supplied past.
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        frequencies = positions[:, None] * self.inv_freq.float()[None]
        angles = torch.cat((frequencies, frequencies), dim=-1)[None, None]
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * dim, dim))
        self.dropout = dropout
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, rotary_cos_sin, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=-1))
        norm = self.norm1(x) * (1 + scale_a) + shift_a
        b, length, dim = x.shape
        qkv = self.attn_qkv(norm).reshape(b, length, 3, self.n_heads, dim // self.n_heads)
        q, k, v = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        cos, sin = rotary_cos_sin
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # Bidirectional prefix + current attention. Past KV cannot be reused unchanged.
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=(self.dropout if self.training else 0.),
                                              is_causal=False)
        attn = attn.transpose(1, 2).reshape(b, length, dim)
        x = x + gate_a * F.dropout(self.attn_out(attn), p=self.dropout, training=self.training)
        hidden = self.norm2(x) * (1 + scale_m) + shift_m
        return x + gate_m * F.dropout(self.mlp(hidden), p=self.dropout, training=self.training)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        if x.is_floating_point():
            return x.to(self.embedding.dtype) @ self.embedding
        return F.embedding(x, self.embedding)


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, vocab_size, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size)
        for layer in (self.linear, self.adaLN_modulation):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=-1)
        return self.linear(self.norm_final(x) * (1 + scale) + shift)


class DIT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_embed = EmbeddingLayer(config.hidden_size, config.codebook_size + 1)
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.rotary_emb = Rotary(config.hidden_size // config.n_heads)
        self.blocks = nn.ModuleList(DDiTBlock(config.hidden_size, config.n_heads, config.cond_dim,
                                             config.dropout) for _ in range(config.n_blocks))
        self.output_layer = DDitFinalLayer(config.hidden_size, config.codebook_size + 1, config.cond_dim)
        self.activation_checkpointing = config.activation_checkpointing

    def forward(self, current, past, sigma):
        embeddings = torch.cat((self.vocab_embed(past), self.vocab_embed(current)), dim=1)
        c = F.silu(self.sigma_map(sigma))
        rotary = self.rotary_emb(embeddings)
        x = embeddings
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, rotary, c, use_reentrant=False)
            else:
                x = block(x, rotary, c)
        # Final normalization/modulation/projection are positionwise; computing
        # only current logits is exactly equivalent to projecting then slicing.
        return self.output_layer(x[:, past.shape[1]:], c)
