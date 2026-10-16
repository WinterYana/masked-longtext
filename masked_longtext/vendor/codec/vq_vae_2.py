# Adapted from masked_reference commit 3ff606831e48be8069cb42a2afffa666051d53fa; see CODEC_NOTES.md.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import NamedTuple, Tuple



def _compute_semantic_losses(
    model,
    z_q_down: torch.Tensor,
    teacher_sem,
    sem_weight: float,
    nbr_weight: float,
    nbr_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute optional semantic alignment and neighbour contrastive losses.

    Both losses are zero when the corresponding weight is 0 or teacher_sem is None.

    Parameters
    ----------
    model : VQ_VAE_2 or VQ_VAE_3
        The model instance (must have ``use_semantic_head``, ``enable_nbr_loss``,
        ``semantic_head``, and ``pool_latent`` attributes).
    z_q_down : Tensor [B, L/2, D]
        Bottom-level quantized latent.
    teacher_sem : Tensor [B, D] or None
        Semantic embedding from the frozen teacher model.
    sem_weight, nbr_weight : float
        Loss weights; losses are skipped when weight == 0.
    nbr_margin : float
        Margin for the triplet-style neighbour contrastive loss.

    Returns
    -------
    sem_loss : Tensor, scalar
    nbr_loss : Tensor, scalar
    """
    device = z_q_down.device
    sem_loss = torch.tensor(0.0, device=device)
    nbr_loss = torch.tensor(0.0, device=device)

    if model.use_semantic_head and (teacher_sem is not None) and float(sem_weight) > 0.0:
        z_pool = model.pool_latent(z_q_down)
        pred_sem = model.semantic_head(z_pool)
        sem_loss = 1.0 - F.cosine_similarity(pred_sem, teacher_sem, dim=-1).mean()

    if model.enable_nbr_loss and (teacher_sem is not None) and float(nbr_weight) > 0.0:
        with torch.no_grad():
            t_norm = F.normalize(teacher_sem, dim=-1)
            sim = t_norm @ t_norm.t()
            sim.fill_diagonal_(-1e9)
            pos_idx = sim.argmax(dim=1)
        z_pool = model.pool_latent(z_q_down)
        z_norm = F.normalize(z_pool, dim=-1)
        pos_sim = (z_norm * z_norm[pos_idx]).sum(dim=-1)
        neg_idx = torch.randperm(z_norm.shape[0], device=z_norm.device)
        neg_sim = (z_norm * z_norm[neg_idx]).sum(dim=-1)
        nbr_loss = F.relu(neg_sim - pos_sim + float(nbr_margin)).mean()

    return sem_loss, nbr_loss


class QuantizerOutput(NamedTuple):
    """Return type of :meth:`Quantizer.find_closest_embeddings`."""
    z_q: torch.Tensor               # quantized latent (same shape as input)
    indices: torch.Tensor           # codebook index per vector, shape [..., 1]
    embedding_loss: torch.Tensor    # codebook loss (zero when use_ema=True)
    commitment_loss: torch.Tensor   # encoder commitment loss
    perplexity: torch.Tensor        # effective number of codes used (scalar)
    alive_codes: int                # number of codes chosen at least once


class VQ_VAE_2_EncodeOutput(NamedTuple):
    """Return type of :meth:`VQ_VAE_2.encode`.

    The 2-level codec compresses a token sequence in two steps::

        tokens [B, L]
          └─ encoder_down (2×) → z_e_down [B, L/2, D] → quantize → z_q_down / idx_down
               └─ encoder_top (2×) → z_e_top [B, L/4, D] → quantize → z_q_top / idx_top
                    └─ decoder_top (2×) → recon_top [B, L/2, D]
                         └─ decoder_down (2×) → token logits [B, L, vocab_size]

    ``idx_top`` (shape [B, L/4]) is the compact discrete code sequence that
    MDLM learns to model as a prior.  Example: 64 tokens → 16 top codes.
    """
    z_q_down: torch.Tensor              # bottom-level quantized latent, shape [B, L/2, D]
    z_q_top: torch.Tensor               # top-level quantized latent,    shape [B, L/4, D]  ← most compressed
    z_e_down: torch.Tensor              # encoder[0] continuous output before quantization
    recon_top: torch.Tensor             # decoder_top output, used as input to decoder_down
    embedding_loss: torch.Tensor        # codebook loss, summed across both levels
    commitment_loss: torch.Tensor       # encoder commitment loss, summed
    perplexity_down: torch.Tensor       # effective # codes used at bottom level (scalar)
    perplexity_top: torch.Tensor        # effective # codes used at top level   (scalar)
    alive_codes_down: int               # number of distinct codes chosen at bottom level
    alive_codes_top: int                # number of distinct codes chosen at top level
    min_encoding_indices_down: torch.Tensor  # discrete code ids, shape [B, L/2, 1]
    min_encoding_indices_top: torch.Tensor   # discrete code ids, shape [B, L/4, 1]  ← MDLM input


class VQ_VAE_2_ForwardOutput(NamedTuple):
    """Return type of :meth:`VQ_VAE_2.forward`."""
    reconstructed_data: torch.Tensor   # token logits [B, T, vocab_size]
    loss: torch.Tensor                 # total training loss (scalar)
    reconstructed_loss: torch.Tensor   # cross-entropy token reconstruction loss
    embedding_loss: torch.Tensor       # VQ codebook loss
    commitment_loss: torch.Tensor      # encoder commitment loss
    perplexity_down: torch.Tensor      # codebook diversity at bottom level
    perplexity_top: torch.Tensor       # codebook diversity at top level
    alive_codes_down: int              # active entries in bottom codebook
    alive_codes_top: int               # active entries in top codebook


class Quantizer(nn.Module):
  """Vector Quantization codebook module.

  Maintains a table of `count_of_codes` learned embedding vectors (the codebook).
  During forward, each input vector is replaced by its nearest codebook entry —
  this forces the model to compress continuous representations into a discrete
  vocabulary of size `count_of_codes`.

  Parameters
  ----------
  count_of_codes : int
      Number of entries in the codebook (vocabulary size of the latent space).
  codes_length : int
      Dimensionality of each codebook vector.
  use_ema : bool
      If True, update codebook via Exponential Moving Average (EMA) of assigned
      input vectors instead of gradient descent. EMA is more stable and prevents
      the codebook from competing with the encoder during backprop.
  beta : float
      Weight of the commitment loss — how strongly the encoder is pushed to stay
      close to the quantized value it has been assigned.
  ema_decay : float
      Decay rate for EMA updates (typically 0.99). Higher = slower update.
  dead_code_check_every : int
      Reinitialize unused codebook entries every this many gradient steps.
      Prevents "code collapse" where only a few entries are ever used.
  dead_code_warmup_updates : int
      Don't check for dead codes until this many steps have passed (gives the
      codebook time to settle before reinitializing).
  dead_code_threshold : float
      A code is considered dead if its EMA usage falls below this fraction.
  dead_code_max_reinit : int
      Maximum number of dead codes to reinitialize in a single check pass.
  """

  def __init__(
      self,
      count_of_codes=512,
      codes_length=1024,
      use_ema=False,
      beta=0.25,
      ema_decay=0.99,
      *,
      dead_code_check_every: int = 1000,
      dead_code_warmup_updates: int = 1000,
      dead_code_threshold: float = 1e-4,
      dead_code_max_reinit: int = 32,
  ):
    super(Quantizer, self).__init__()
    self.count_of_codes = count_of_codes
    self.codes_length = codes_length
    self.code_embeddings = nn.Embedding(num_embeddings=self.count_of_codes, embedding_dim=self.codes_length)
    # Uniform initialization scaled by 1/sqrt(D) — the standard
    # "Xavier uniform" range that keeps initial embedding norms ~1.
    init_scale = 1.0 / (self.codes_length ** 0.5)
    self.code_embeddings.weight.data.uniform_(-init_scale, init_scale)
    self.use_ema = use_ema
    self.beta = beta
    self.ema_decay = ema_decay

    self.dead_code_check_every = int(dead_code_check_every)
    self.dead_code_warmup_updates = int(dead_code_warmup_updates)
    self.dead_code_threshold = float(dead_code_threshold)
    self.dead_code_max_reinit = int(dead_code_max_reinit)

    if self.use_ema:
      self.register_buffer('ema_cluster_size', torch.zeros(self.count_of_codes))
      self.register_buffer('ema_w', torch.zeros(self.count_of_codes, self.codes_length))
      self.ema_w.data.copy_(self.code_embeddings.weight.data)
      # EMA-updated codebooks should not receive optimizer gradients.
      # requires_grad_ is not supported on all parameter types; ignore if it fails.
      try:
        self.code_embeddings.weight.requires_grad_(False)
      except AttributeError:
        pass

    self.register_buffer('code_usage_count', torch.zeros(count_of_codes))
    self.register_buffer('total_updates', torch.tensor(0))

  def _dist_is_ready(self) -> bool:
    return bool(dist.is_available() and dist.is_initialized())

  def _all_reduce_(self, x: torch.Tensor) -> torch.Tensor:
    if self._dist_is_ready():
      dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

  # ------------------------------------------------------------------
  # Public entry point
  # ------------------------------------------------------------------

  def find_closest_embeddings(self, z, current_beta=None):
    """Quantize ``z`` by replacing each vector with its nearest codebook entry.

    Orchestrates four sub-steps (each implemented as a private method below):

    1. :meth:`_maybe_init_codebook` — seed codebook from data on the very first batch.
    2. :meth:`_lookup_nearest`      — compute squared-L2 distances and pick nearest code.
    3. :meth:`_update_codebook_ema` — move codes toward their assigned inputs via EMA.
    4. :meth:`_reinitialize_dead_codes` — periodically replace unused codes.

    Parameters
    ----------
    z : Tensor, shape [B, T, D] or [N, D]
        Input latent vectors to quantize.
    current_beta : float, optional
        Commitment loss weight for this step. Defaults to ``self.beta``.

    Returns
    -------
    QuantizerOutput with fields:
        z_q : Tensor, same shape as z
            Quantized latents with straight-through gradient.
        indices : Tensor, shape [..., 1]
            Integer codebook index for each input vector.
        embedding_loss : Tensor, scalar
            Codebook loss (zero when use_ema=True).
        commitment_loss : Tensor, scalar
            Encoder commitment loss.
        perplexity : Tensor, scalar
            Effective number of codes used this batch.
        alive_codes : int
            Number of codes chosen at least once.
    """
    if current_beta is None:
        current_beta = self.beta

    # Flatten to (N, D) for distance computations, remember original shape.
    z_flattened = z.reshape(-1, self.codes_length)
    with torch.autocast(device_type=z.device.type, enabled=False):
        z_flattened = z_flattened.float()
    n_assign = int(z_flattened.shape[0])

    # Step 1: Initialize codebook from data on the very first batch.
    if self.training and int(self.total_updates.item()) == 0:
        self._maybe_init_codebook(z_flattened, n_assign)

    # Step 2: Find the nearest codebook entry for each input vector.
    min_encoding_indices = self._lookup_nearest(z_flattened)

    # Per-code assignment counts for this batch (used by steps 3 & 4).
    counts = torch.bincount(min_encoding_indices, minlength=self.count_of_codes).to(torch.float32)
    counts = self._all_reduce_(counts)
    n_total = float(counts.sum().item()) or float(max(1, n_assign))

    # Track usage EMA (probability scale, so values stay in [0, 1]).
    if self.training:
        p_batch = counts / n_total
        self.code_usage_count.mul_(self.ema_decay).add_(p_batch, alpha=(1.0 - self.ema_decay))
        self.total_updates += 1

    # Look up quantized embeddings and reshape back to z's original shape.
    z_q = self.code_embeddings(min_encoding_indices).view(z.shape)

    # Step 3: Update codebook entries via EMA (only during training, use_ema mode).
    if self.training and self.use_ema:
        self._update_codebook_ema(z_flattened, min_encoding_indices, counts)

    # Step 4: Reinitialize dead codes periodically.
    if (
        self.training
        and self.dead_code_check_every > 0
        and int(self.total_updates.item()) >= self.dead_code_warmup_updates
        and int(self.total_updates.item()) % int(self.dead_code_check_every) == 0
    ):
        self._reinitialize_dead_codes(z_flattened)

    # Losses and straight-through estimator.
    if self.use_ema:
        embedding_loss = torch.zeros((), device=z.device, dtype=z.dtype)
    else:
        embedding_loss = torch.mean((z.detach() - z_q) ** 2)
    commitment_loss = current_beta * torch.mean((z - z_q.detach()) ** 2)
    # Straight-through: forward = z_q, backward = z (gradient passes through).
    z_q = z + (z_q - z).detach()

    p = counts / n_total
    perplexity = torch.exp(-torch.sum(p * torch.log(p + 1e-10)))
    alive_codes = int((counts > 0).sum().item())

    return QuantizerOutput(z_q, min_encoding_indices.unsqueeze(1), embedding_loss, commitment_loss, perplexity, alive_codes)

  # ------------------------------------------------------------------
  # Private helpers (called only from find_closest_embeddings)
  # ------------------------------------------------------------------

  def _maybe_init_codebook(self, z_flattened: torch.Tensor, n_assign: int) -> None:
    """Seed codebook entries from input data on the very first training batch.

    By default, codebook weights are randomly initialized which can cause all
    inputs to collapse to a single code early in training.  This method copies
    ``count_of_codes`` random input vectors into the codebook so every entry
    starts near the actual data distribution.

    Only rank-0 performs the write in distributed training; it then broadcasts
    the result so all ranks start with the same codebook.
    """
    usage_sum = float(self.code_usage_count.detach().sum().item())
    ema_sum = float(self.ema_cluster_size.detach().sum().item()) if self.use_ema else 0.0
    if usage_sum != 0.0 or (self.use_ema and ema_sum != 0.0) or n_assign <= 0:
        return

    dist_ready = self._dist_is_ready()
    if (not dist_ready) or dist.get_rank() == 0:
        with torch.no_grad():
            k = int(self.count_of_codes)
            perm = torch.randperm(n_assign, device=z_flattened.device)
            if n_assign >= k:
                pick = perm[:k]
            else:
                # If batch is smaller than codebook, repeat indices.
                reps = (k + n_assign - 1) // n_assign
                pick = perm.repeat(reps)[:k]
            vals = z_flattened[pick].detach().to(self.code_embeddings.weight.dtype)
            self.code_embeddings.weight.data.copy_(vals)
            if self.use_ema:
                self.ema_w.data.copy_(vals.to(self.ema_w.dtype))
                self.ema_cluster_size.zero_().add_(1.0)
    if dist_ready:
        dist.broadcast(self.code_embeddings.weight.data, src=0)
        if self.use_ema:
            dist.broadcast(self.ema_w, src=0)
            dist.broadcast(self.ema_cluster_size, src=0)

  def _lookup_nearest(self, z_flattened: torch.Tensor) -> torch.Tensor:
    """Return the codebook index closest to each row of ``z_flattened``.

    Uses the identity  ||z - e||² = ||z||² + ||e||² - 2·z·eᵀ  to avoid
    materialising a full (N, D) minus broadcast — more numerically stable
    and faster than a naive loop.

    Args:
        z_flattened: Float tensor of shape (N, D).

    Returns:
        Integer index tensor of shape (N,).
    """
    with torch.autocast(device_type=z_flattened.device.type, enabled=False):
        z_f = z_flattened.float()
        e_f = self.code_embeddings.weight.float()
        z2 = (z_f * z_f).sum(dim=1, keepdim=True)   # (N, 1)
        e2 = (e_f * e_f).sum(dim=1).unsqueeze(0)     # (1, K)
        distances = z2 + e2 - 2.0 * (z_f @ e_f.t()) # (N, K)
        return torch.argmin(distances, dim=1)         # (N,)

  def _update_codebook_ema(
      self,
      z_flattened: torch.Tensor,
      min_encoding_indices: torch.Tensor,
      counts: torch.Tensor,
  ) -> None:
    """Update codebook entries toward their assigned inputs via EMA.

    For each code ``k`` with assignment count ``n_k``:
      ema_cluster_size[k] = decay * ema_cluster_size[k] + (1-decay) * n_k
      ema_w[k]            = decay * ema_w[k]            + (1-decay) * sum(z assigned to k)
      codebook[k]         = ema_w[k] / laplace_smoothed(ema_cluster_size[k])

    This is equivalent to an online k-means update and is done without
    gradients — the encoder gradients come only from the commitment loss.

    Args:
        z_flattened:         Float tensor (N, D).
        min_encoding_indices: Integer tensor (N,) with codebook indices.
        counts:              Float tensor (K,) with per-code assignment counts.
    """
    with torch.no_grad():
        cluster_update = counts.to(self.ema_cluster_size.dtype)
        # Sum of all input vectors assigned to each code.
        dw = torch.zeros(
            self.count_of_codes, self.codes_length,
            device=z_flattened.device, dtype=torch.float32,
        )
        dw.index_add_(0, min_encoding_indices, z_flattened)
        dw = self._all_reduce_(dw)

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_update, alpha=1 - self.ema_decay)
        self.ema_w.mul_(self.ema_decay).add_(dw.to(self.ema_w.dtype), alpha=1 - self.ema_decay)

        # Laplace-smooth cluster sizes to avoid division-by-zero.
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.count_of_codes * 1e-5) * n
        embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
        self.code_embeddings.weight.data.copy_(embed_normalized)

  def _reinitialize_dead_codes(self, z_flattened):
    """Reinitialize codes that haven't been used recently"""
    thr = float(self.dead_code_threshold)
    if thr <= 0:
      return
    dead_mask = self.code_usage_count < thr
    n_dead = int(dead_mask.sum().item())

    if n_dead <= 0:
      return

    # In distributed training, only rank-0 mutates the codebook, then broadcasts.
    dist_ready = self._dist_is_ready()
    max_reinit = int(self.dead_code_max_reinit) if int(self.dead_code_max_reinit) > 0 else n_dead
    k = min(n_dead, max_reinit, int(z_flattened.shape[0]))
    if k <= 0:
      return

    dead_indices_all = torch.where(dead_mask)[0]
    # Reinit the "most dead" codes first (smallest usage EMA).
    usage = self.code_usage_count[dead_indices_all]
    order = torch.argsort(usage)
    dead_indices = dead_indices_all[order[:k]]

    if (not dist_ready) or dist.get_rank() == 0:
      random_indices = torch.randperm(int(z_flattened.shape[0]), device=z_flattened.device)[:k]
      new_vals = z_flattened[random_indices].detach().to(self.code_embeddings.weight.dtype)
      self.code_embeddings.weight.data[dead_indices] = new_vals
      # Prevent immediate reinit loops.
      self.code_usage_count[dead_indices] = float(max(thr, 1e-6))

      if self.use_ema:
        self.ema_w[dead_indices] = new_vals.to(self.ema_w.dtype)
        self.ema_cluster_size[dead_indices] = 1.0

    if dist_ready:
      # Keep all ranks in sync.
      dist.broadcast(self.code_embeddings.weight.data, src=0)
      dist.broadcast(self.code_usage_count, src=0)
      if self.use_ema:
        dist.broadcast(self.ema_w, src=0)
        dist.broadcast(self.ema_cluster_size, src=0)







def _build_semantic_head(codes_length: int, sem_proj_dim: int, semantic_dim: int) -> nn.Sequential:
    """Build the MLP that predicts teacher semantic embeddings from quantized latents."""
    return nn.Sequential(
        nn.LayerNorm(codes_length),
        nn.Linear(codes_length, sem_proj_dim),
        nn.GELU(),
        nn.Linear(sem_proj_dim, semantic_dim),
    )


def pool_latent(z: torch.Tensor) -> torch.Tensor:
    """Mean-pool a [B, T, D] latent sequence to [B, D]. Pass-through for 2-D inputs."""
    if z.dim() == 3:
        return z.mean(dim=1)
    return z


class VQ_VAE_2(nn.Module):
  def __init__(self, encoders, decoders, top_quantizer_count_of_codes=512,
               down_quantizer_count_of_codes=512, codes_length=512, beta=0.25,
               use_ema=False, ema_decay=0.99, top_recon_loss_weight=0.2,
               *, use_semantic_head: bool = True, semantic_dim: int = 768, sem_proj_dim: int = 512, enable_nbr_loss: bool = True):
    super(VQ_VAE_2, self).__init__()

    self.beta = beta
    self.use_ema = use_ema
    self.ema_decay = ema_decay
    self.codes_length = codes_length
    self.top_recon_loss_weight = top_recon_loss_weight
    # Use nn.ModuleList so PyTorch tracks submodule parameters for optimization.
    self.encoders = nn.ModuleList(encoders)
    self.decoders = nn.ModuleList(decoders)

    self.top_quantizer = Quantizer(count_of_codes=top_quantizer_count_of_codes,
                                   codes_length=codes_length,
                                   use_ema=use_ema,
                                   beta=beta,
                                   ema_decay=ema_decay)
    self.down_quantizer = Quantizer(count_of_codes=down_quantizer_count_of_codes,
                                   codes_length=codes_length,
                                   use_ema=use_ema,
                                   beta=beta,
                                   ema_decay=ema_decay)
    # --- Semantic bottleneck (teacher-embedding prediction) ---
    self.use_semantic_head = bool(use_semantic_head)
    self.semantic_dim = int(semantic_dim)
    self.enable_nbr_loss = bool(enable_nbr_loss)
    if self.use_semantic_head:
      # Predict teacher semantic embedding from pooled quantized latent.
      self.semantic_head = _build_semantic_head(self.codes_length, int(sem_proj_dim), int(semantic_dim))
    else:
      self.semantic_head = None

    # Buffers for logging (set in forward).
    self.last_sem_loss = torch.tensor(0.0)
    self.last_nbr_loss = torch.tensor(0.0)
  def encode(self, t, X, current_beta=None):
    """Run the two-level encoding pipeline and quantize at each level.

    Pipeline: tokens -> encoder[0] -> encoder[1] -> top_quantizer
              -> decoder[0] (intermediate) -> down_quantizer

    Parameters
    ----------
    t : Tensor
        Timestep or noise-level embeddings, shape [B, T, D] or [B, D].
    X : Tensor
        Input token embeddings or one-hot vectors.
    current_beta : float, optional
        Commitment loss weight override. Defaults to ``self.beta``.

    Returns (12-tuple, in order)
    -----------------------------
    z_q_down : Tensor        -- bottom-level quantized latent
    z_q_top : Tensor         -- top-level quantized latent
    z_e_down : Tensor        -- encoder[0] output before quantization
    recon_top : Tensor       -- intermediate decoder output from top stage
    embedding_loss : Tensor  -- codebook loss, summed across both levels
    commitment_loss : Tensor -- encoder commitment loss, summed across both levels
    perplexity_down : Tensor -- codebook usage perplexity at bottom level
    perplexity_top : Tensor  -- codebook usage perplexity at top level
    alive_codes_down : int   -- number of active bottom-level codebook entries
    alive_codes_top : int    -- number of active top-level codebook entries
    min_encoding_indices_down : Tensor  -- discrete code indices (bottom level)
    min_encoding_indices_top : Tensor   -- discrete code indices (top level)
    """
    z_e_down = self.encoders[0](t, X)
    z_e_top = self.encoders[1](t, z_e_down)

    top = self.top_quantizer.find_closest_embeddings(z_e_top, current_beta)
    recon_top = self.decoders[0](t, top.z_q, x_from_previous_stage=None)
    down = self.down_quantizer.find_closest_embeddings(recon_top, current_beta)
    z_q_top, min_encoding_indices_top, embedding_loss_top, commitment_loss_top, perplexity_top, alive_codes_top = top
    z_q_down, min_encoding_indices_down, embedding_loss_down, commitment_loss_down, perplexity_down, alive_codes_down = down
    embedding_loss = embedding_loss_down + embedding_loss_top
    commitment_loss = commitment_loss_down + commitment_loss_top

    return VQ_VAE_2_EncodeOutput(
        z_q_down=z_q_down,
        z_q_top=z_q_top,
        z_e_down=z_e_down,
        recon_top=recon_top,
        embedding_loss=embedding_loss,
        commitment_loss=commitment_loss,
        perplexity_down=perplexity_down,
        perplexity_top=perplexity_top,
        alive_codes_down=alive_codes_down,
        alive_codes_top=alive_codes_top,
        min_encoding_indices_down=min_encoding_indices_down,
        min_encoding_indices_top=min_encoding_indices_top,
    )

  def decode(self, t, z_q_down, z_q_top):
    recon_down = self.decoders[1](t, z_q_down, x_from_previous_stage=z_q_top)
    return recon_down

  def decode_from_codes(self, t: torch.Tensor, top_codes: torch.Tensor) -> torch.Tensor:
    """Decode token logits from top-level discrete code indices.

    This is the inverse of encode() for inference: given the compact top-level
    codes (e.g. produced by MDLM), reconstruct the full-resolution token logits.

    Parameters
    ----------
    t : Tensor [B]
        Timestep vector (typically ``torch.ones(B)``).
    top_codes : Tensor [B, L/4] or [B, L/4, 1]
        Integer indices into the top codebook.

    Returns
    -------
    logits : Tensor [B, L, vocab_size]
    """
    if top_codes.ndim == 3:
        top_codes = top_codes.squeeze(-1)
    z_q_top = self.top_quantizer.code_embeddings(top_codes)
    recon_top = self.decoders[0](t, z_q_top, x_from_previous_stage=None)
    z_q_down, *_ = self.down_quantizer.find_closest_embeddings(recon_top, None)
    return self.decoders[1](t, z_q_down, x_from_previous_stage=z_q_top)

  def pool_latent(self, z_q_down: torch.Tensor) -> torch.Tensor:
    return pool_latent(z_q_down)

  def forward(
    self,
    t,
    X,
    input_tokens,
    current_beta=None,
    *,
    teacher_sem=None,
    sem_weight: float = 0.0,
    nbr_weight: float = 0.0,
    nbr_margin: float = 0.2,
  ):
    """Full encode -> decode -> loss computation.

    Parameters
    ----------
    t : Tensor
        Timestep embeddings.
    X : Tensor
        Input token embeddings.
    input_tokens : Tensor, shape [B, T]
        Ground-truth token ids used to compute reconstruction loss.
    current_beta : float, optional
        Commitment loss weight for this step.
    teacher_sem : Tensor, optional
        Semantic embeddings from a frozen teacher model (e.g. GPT-2).
        Used to compute the auxiliary semantic alignment loss.
    sem_weight : float
        Weight for the semantic cosine-similarity loss.
    nbr_weight : float
        Weight for the neighbor contrastive loss.
    nbr_margin : float
        Margin for the neighbor loss triplet.

    Returns (9-tuple, in order)
    ----------------------------
    reconstructed_data : Tensor  -- token logits [B, T, vocab_size]
    loss : Tensor                -- total training loss (scalar)
    reconstructed_loss : Tensor  -- cross-entropy token reconstruction loss
    embedding_loss : Tensor      -- VQ codebook loss
    commitment_loss : Tensor     -- encoder commitment loss
    perplexity_down : Tensor     -- codebook diversity at bottom level
    perplexity_top : Tensor      -- codebook diversity at top level
    alive_codes_down : int       -- active entries in bottom codebook
    alive_codes_top : int        -- active entries in top codebook
    """
    enc = self.encode(t, X, current_beta)
    z_q_down = enc.z_q_down
    z_q_top = enc.z_q_top
    z_e_down = enc.z_e_down
    recon_top = enc.recon_top
    embedding_loss = enc.embedding_loss
    commitment_loss = enc.commitment_loss
    perplexity_down = enc.perplexity_down
    perplexity_top = enc.perplexity_top
    alive_codes_down = enc.alive_codes_down
    alive_codes_top = enc.alive_codes_top

    reconstructed_data = self.decode(t, z_q_down, z_q_top)
    reconstructed_loss = F.cross_entropy(
      reconstructed_data.view(-1, reconstructed_data.shape[-1]),
      input_tokens.view(-1),
      reduction="mean",
    )
    top_reconstruction_loss = F.mse_loss(recon_top, z_e_down.detach())

    # --- Semantic bottleneck losses (optional) ---
    sem_loss, nbr_loss = _compute_semantic_losses(
        self, z_q_down, teacher_sem, sem_weight, nbr_weight, nbr_margin
    )

    loss = (
      reconstructed_loss
      + embedding_loss
      + commitment_loss
      + self.top_recon_loss_weight * top_reconstruction_loss
      + float(sem_weight) * sem_loss
      + float(nbr_weight) * nbr_loss
    )

    # Store for logging.
    self.last_sem_loss = sem_loss.detach()
    self.last_nbr_loss = nbr_loss.detach()

    return VQ_VAE_2_ForwardOutput(
      reconstructed_data,
      loss,
      reconstructed_loss,
      embedding_loss,
      commitment_loss,
      perplexity_down,
      perplexity_top,
      alive_codes_down,
      alive_codes_top,
    )
