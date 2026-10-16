"""Portable two-level text VQ codec adapted from the selected reference commit."""
from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .vendor.codec.gpt_blocks import GPT, GPTConfig
from .vendor.codec.vq_vae_2 import VQ_VAE_2


@dataclass(frozen=True)
class CodecConfig:
    """Explicit restored settings, not a claim of historical checkpoint identity."""
    vocab_size: int
    codebook_size: int = 512
    dim: int = 384
    num_heads: int = 6
    num_layers: int = 3
    beta: float = 0.05
    use_ema: bool = True
    ema_decay: float = 0.99
    top_recon_loss_weight: float = 0.2
    conv_dropout: float = 0.1
    dead_code_check_every: int = 1000
    dead_code_warmup_updates: int = 1000
    dead_code_threshold: float = 1e-4
    dead_code_max_reinit: int = 32
    logit_chunk_size: int = 256
    checkpoint_reconstruction: bool = True

    @property
    def compression_factor(self):
        return 4

    def __post_init__(self):
        for name in ("vocab_size", "codebook_size", "dim", "num_heads", "num_layers"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.num_heads or (self.dim // self.num_heads) % 2:
            raise ValueError("num_heads must divide dim with an even rotary head dimension")
        if self.num_layers % 2 != 1:
            raise ValueError("num_layers must be odd for the middle compression block")
        for name in ("dead_code_check_every", "dead_code_warmup_updates", "dead_code_max_reinit", "logit_chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("beta", "top_recon_loss_weight", "dead_code_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if not math.isfinite(self.conv_dropout) or not 0 <= self.conv_dropout < 1:
            raise ValueError("conv_dropout must be in [0, 1)")
        if type(self.use_ema) is not bool or type(self.checkpoint_reconstruction) is not bool:
            raise TypeError("use_ema and checkpoint_reconstruction must be booleans")


class TextCodec(nn.Module):
    """VQ-VAE-2 with full contiguous token sequences and a fourfold top bottleneck.

    TextCodec.forward trains the codebook exactly once at each of the two levels.
    encode_codes/decode_codes are state-preserving, no-grad inference methods even
    when the caller left this module in training mode. No padding token is implied.
    """
    def __init__(self, config: CodecConfig):
        super().__init__()
        if not isinstance(config, CodecConfig):
            raise TypeError("config must be CodecConfig")
        self.config = config

        def stage(input_dim, output_dim, *, compress=False, decompress=False, fuse=False, text=False):
            model = GPT(GPTConfig(vocab_size=input_dim, n_layer=config.num_layers,
                n_head=config.num_heads, n_embd=config.dim, output_dim=output_dim,
                conv_dropout=config.conv_dropout), is_compress=compress,
                is_decompress=decompress, need_fuse=fuse)
            if not text:
                model.transformer.wte = nn.Identity()
            return model

        encoders = [stage(config.vocab_size, config.dim, compress=True, text=True),
                    stage(config.dim, config.dim, compress=True)]
        decoders = [stage(config.dim, config.dim, decompress=True),
                    stage(config.dim, config.vocab_size, decompress=True, fuse=True)]
        self.model = VQ_VAE_2(encoders, decoders,
            top_quantizer_count_of_codes=config.codebook_size,
            down_quantizer_count_of_codes=config.codebook_size,
            codes_length=config.dim, beta=config.beta, use_ema=config.use_ema,
            ema_decay=config.ema_decay, top_recon_loss_weight=config.top_recon_loss_weight,
            use_semantic_head=False, enable_nbr_loss=False)
        for quantizer in (self.model.top_quantizer, self.model.down_quantizer):
            for name in ("dead_code_check_every", "dead_code_warmup_updates", "dead_code_threshold", "dead_code_max_reinit"):
                setattr(quantizer, name, getattr(config, name))

    def _validate_ids(self, ids, *, name, upper_bound, multiple=1):
        if not isinstance(ids, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if ids.dtype != torch.long:
            raise TypeError(f"{name} must have dtype torch.long")
        if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] == 0:
            raise ValueError(f"{name} must have nonempty shape [batch, length]")
        if ids.shape[1] % multiple:
            raise ValueError(f"{name} length must be divisible by {multiple}")
        if bool((ids < 0).any()) or bool((ids >= upper_bound).any()):
            raise ValueError(f"{name} IDs must be in [0, {upper_bound}); MASK/PAD is unsupported")
        if ids.device != next(self.parameters()).device:
            raise ValueError(f"{name} and codec parameters must be on the same device")

    @staticmethod
    def _time(ids):
        # The reference extraction/decoding path conditions the codec on t=1.
        return torch.ones(ids.shape[0], device=ids.device, dtype=torch.float32)

    @contextmanager
    def _inference(self):
        modules = list(self.modules())
        modes = [module.training for module in modules]
        try:
            self.eval()
            with torch.no_grad():
                yield
        finally:
            for module, mode in zip(modules, modes):
                module.training = mode

    def _head_ce(self, hidden, conditioning, targets):
        logits = self.model.decoders[1].final_layer(hidden, conditioning).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")

    def _reconstruction_ce(self, hidden, conditioning, tokens):
        chunk = self.config.logit_chunk_size or tokens.shape[1]
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, tokens.shape[1], chunk):
            args = (hidden[:, start:start + chunk], conditioning, tokens[:, start:start + chunk])
            if self.config.checkpoint_reconstruction and torch.is_grad_enabled():
                loss = checkpoint(self._head_ce, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                loss = self._head_ce(*args)
            total = total + loss
        return total / tokens.numel()

    def forward(self, tokens: torch.Tensor):
        self._validate_ids(tokens, name="tokens", upper_bound=self.config.vocab_size, multiple=4)
        t = self._time(tokens)
        enc = self.model.encode(t, tokens)
        hidden, conditioning = self.model.decoders[1].forward_hidden(t, enc.z_q_down, enc.z_q_top)
        ce = self._reconstruction_ce(hidden, conditioning, tokens)
        top_reconstruction_loss = F.mse_loss(enc.recon_top, enc.z_e_down.detach())
        loss = ce + enc.embedding_loss + enc.commitment_loss + self.config.top_recon_loss_weight * top_reconstruction_loss
        return dict(loss=loss, ce=ce, embedding_loss=enc.embedding_loss,
            commitment_loss=enc.commitment_loss, top_reconstruction_loss=top_reconstruction_loss,
            perplexity_down=enc.perplexity_down, perplexity_top=enc.perplexity_top,
            alive_codes_down=ce.new_tensor(enc.alive_codes_down),
            alive_codes_top=ce.new_tensor(enc.alive_codes_top))

    def encode_codes(self, tokens: torch.Tensor):
        self._validate_ids(tokens, name="tokens", upper_bound=self.config.vocab_size, multiple=4)
        with self._inference():
            # Code extraction needs the encoders and top quantizer only. Avoid
            # the intermediate decoder/bottom lookup that cannot affect top IDs.
            t = self._time(tokens)
            bottom = self.model.encoders[0](t, tokens)
            top = self.model.encoders[1](t, bottom)
            result = self.model.top_quantizer.find_closest_embeddings(top)
            return result.indices.reshape(tokens.shape[0], tokens.shape[1] // 4)

    def decode_codes(self, codes: torch.Tensor):
        self._validate_ids(codes, name="codes", upper_bound=self.config.codebook_size)
        with self._inference():
            return self.model.decode_from_codes(self._time(codes), codes)

    def decode_token_ids(self, codes: torch.Tensor):
        """Jointly decode all features, then chunk only the vocabulary head.

        Every attention/convolution/fusion stage sees the full sequence. Only
        the final positionwise vocabulary projection is chunked, so normal
        generation need not allocate [B, 4L, vocab_size] reconstruction logits.
        """
        self._validate_ids(codes, name="codes", upper_bound=self.config.codebook_size)
        with self._inference():
            t = self._time(codes)
            top = self.model.top_quantizer.code_embeddings(codes)
            reconstructed = self.model.decoders[0](t, top, x_from_previous_stage=None)
            bottom = self.model.down_quantizer.find_closest_embeddings(reconstructed).z_q
            decoder = self.model.decoders[1]
            hidden, conditioning = decoder.forward_hidden(t, bottom, top)
            chunk = self.config.logit_chunk_size or hidden.shape[1]
            ids = []
            for start in range(0, hidden.shape[1], chunk):
                logits = decoder.final_layer(hidden[:, start:start + chunk], conditioning).float()
                ids.append(logits.argmax(dim=-1))
            return torch.cat(ids, dim=1)
