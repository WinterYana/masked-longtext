"""Prefix-block MDLM: DiT, log-linear absorbing masking, loss, EMA and sampling.

Derived exclusively from reference commit 3ff606831e48be8069cb42a2afffa666051d53fa.
This portable reconstructed baseline does not claim historical weight compatibility
or measured text-quality gains. Within-block confidence refinement is experimental.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from .vendor.dit import DIT


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_eps(value, name):
    # Below the float64 normal range reciprocal schedule/time weights may
    # overflow or depend on hardware subnormal handling. Reject, never clip.
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not sys.float_info.min <= value < 1):
        raise ValueError(f"{name} must be finite in [{sys.float_info.min}, 1)")


def _time_dtype(config):
    # Keep the default precision and RNG stream; only unusually small epsilons
    # need float64 to retain nonzero endpoints and finite reciprocal weights.
    return (torch.float64 if min(config.sampling_eps, config.noise_eps) < torch.finfo(torch.float32).tiny
            else torch.float32)


@dataclass(frozen=True)
class DenoiserConfig:
    codebook_size: int = 512
    hidden_size: int = 512
    n_blocks: int = 8
    n_heads: int = 8
    cond_dim: int = 128
    dropout: float = .1
    time_conditioning: bool = False
    block_length: int = 16
    activation_checkpointing: bool = False

    def __post_init__(self):
        for field in ("codebook_size", "hidden_size", "n_blocks", "n_heads", "cond_dim", "block_length"):
            _positive_int(getattr(self, field), field)
        if self.hidden_size % self.n_heads or (self.hidden_size // self.n_heads) % 2:
            raise ValueError("hidden_size must divide into even-sized attention heads for rotary embeddings")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class LossConfig:
    sampling_eps: float = 1e-5
    noise_eps: float = 1e-3
    antithetic_sampling: bool = True
    past_code_corruption_p: float = .1
    past_label_smoothing: float = 0.

    def __post_init__(self):
        _validate_eps(self.sampling_eps, "sampling_eps")
        _validate_eps(self.noise_eps, "noise_eps")
        for field in ("past_code_corruption_p", "past_label_smoothing"):
            if not 0 <= getattr(self, field) <= 1:
                raise ValueError(f"{field} must be in [0, 1]")


@dataclass(frozen=True)
class SamplingConfig:
    steps: int = 128
    predictor: str = "ddpm_cache"
    temperature: float = 1.
    candidates: int = 1
    score_draws: int = 16
    score_formula: str = "target_length"
    sampling_eps: float = 1e-5
    noise_eps: float = 1e-3
    max_context_codes: int | None = None
    refinement_rounds: int = 0
    refinement_fraction: float = .25
    refinement_steps: int = 32
    past_mix_alpha: float = 1.

    def __post_init__(self):
        for field in ("steps", "candidates", "score_draws", "refinement_steps"):
            _positive_int(getattr(self, field), field)
        if self.max_context_codes is not None:
            _positive_int(self.max_context_codes, "max_context_codes")
        if isinstance(self.refinement_rounds, bool) or not isinstance(self.refinement_rounds, int) or self.refinement_rounds < 0:
            raise ValueError("refinement_rounds must be a nonnegative integer")
        if self.predictor not in ("ddpm", "ddpm_cache"):
            raise ValueError("predictor must be ddpm or ddpm_cache")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.score_formula not in ("target_length", "legacy_mask_count"):
            raise ValueError("score_formula must be target_length or legacy_mask_count")
        _validate_eps(self.sampling_eps, "sampling_eps")
        _validate_eps(self.noise_eps, "noise_eps")
        if not 0 < self.refinement_fraction <= 1 or not 0 <= self.past_mix_alpha <= 1:
            raise ValueError("refinement_fraction must be in (0, 1], past_mix_alpha in [0, 1]")


def _validate_codes(codes, codebook_size, *, name, allow_mask=False):
    if not isinstance(codes, torch.Tensor) or codes.ndim != 2 or codes.dtype != torch.long:
        raise ValueError(f"{name} must be a LongTensor [batch, length]")
    bound = codebook_size + int(allow_mask)
    if codes.numel() and (torch.any(codes < 0) or torch.any(codes >= bound)):
        raise ValueError(f"{name} contains codes outside [0, {bound - 1}]")


def _validate_past(past, current, codebook_size):
    if not isinstance(past, torch.Tensor):
        raise ValueError("past must be a tensor")
    if past.ndim == 2:
        _validate_codes(past, codebook_size, name="past", allow_mask=True)
    elif past.ndim == 3 and past.is_floating_point() and past.shape[-1] == codebook_size + 1:
        if not torch.isfinite(past).all() or torch.any(past < 0):
            raise ValueError("soft past must contain finite nonnegative probabilities")
        if past.numel() and not torch.allclose(past.sum(-1), torch.ones_like(past[..., 0]), atol=1e-4, rtol=1e-4):
            raise ValueError("soft past distributions must sum to one")
    else:
        raise ValueError("past must be Long[B,P] or Float[B,P,K+1]")
    if past.shape[0] != current.shape[0] or past.device != current.device:
        raise ValueError("past and current must share batch size and device")


class PrefixDenoiser(nn.Module):
    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config
        self.backbone = DIT(config)

    def forward(self, current, past, sigma):
        _validate_codes(current, self.config.codebook_size, name="current", allow_mask=True)
        if current.shape[0] == 0 or current.shape[1] == 0:
            raise ValueError("current batch and length must be nonempty")
        _validate_past(past, current, self.config.codebook_size)
        if sigma.shape != (current.shape[0],) or not sigma.is_floating_point() or sigma.device != current.device:
            raise ValueError("sigma must be Float[B] on the same device as current")
        if not torch.isfinite(sigma).all() or torch.any(sigma < 0):
            raise ValueError("sigma must be finite and nonnegative")
        if not self.config.time_conditioning:
            sigma = torch.zeros_like(sigma)
        return self.backbone(current, past, sigma)


def loglinear_schedule(t, noise_eps=1e-3):
    """Return sigma(t), sigma'(t); mask probability is (1-noise_eps)*t."""
    _validate_eps(noise_eps, "noise_eps")
    if not t.is_floating_point() or not torch.isfinite(t).all() or torch.any((t < 0) | (t > 1)):
        raise ValueError("t must contain finite floating values in [0, 1]")
    # Promote low-precision inputs, and very small epsilons requiring finite
    # reciprocals. Avoid 1-(1-eps)*t, which rounds to zero at t=1 in float32.
    if t.dtype in (torch.float16, torch.bfloat16):
        t = t.float()
    if noise_eps < torch.finfo(torch.float32).tiny:
        t = t.double()
    factor = 1 - noise_eps
    survival = (1 - t) + noise_eps * t
    # log1p retains small positive sigma where survival itself rounds to one.
    # Its inactive branch receives t=0, avoiding an unselected log1p(-1).
    early = t <= .5
    early_t = torch.where(early, t, torch.zeros_like(t))
    sigma = torch.where(early, -torch.log1p(-factor * early_t), -torch.log(survival))
    return sigma, factor / survival


def substitution_log_probs(logits, current, mask_index, temperature=1.):
    """SUBS predicts regular codes only and fixes every already revealed position."""
    if logits.shape != (*current.shape, mask_index + 1):
        raise ValueError("logits and current shapes disagree")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logp = torch.cat((F.log_softmax(logits[..., :mask_index].float() / temperature, dim=-1),
                      torch.full_like(logits[..., :1].float(), -torch.inf)), dim=-1)
    fixed = torch.full_like(logp, -torch.inf).scatter(-1, current[..., None], 0.)
    return torch.where((current == mask_index)[..., None], logp, fixed)


def _sample_times(batch_size, device, config, generator):
    dtype = _time_dtype(config)
    eps = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    if config.antithetic_sampling:
        eps = (eps / batch_size + torch.arange(batch_size, device=device, dtype=dtype) / batch_size) % 1
    return config.sampling_eps + (1 - config.sampling_eps) * eps


def masked_loss(model, past, current, config: LossConfig, generator=None, validation=False):
    """Mask-only weighted NLL averaged over ALL target positions, with clean validation past.

    The absorbing log-linear weight sigma'/expm1(sigma) is exactly 1/t.
    Only regular code IDs may replace past codes; smoothing gives MASK zero mass.
    """
    k = model.config.codebook_size
    _validate_codes(current, k, name="current")
    _validate_codes(past, k, name="past")
    _validate_past(past, current, k)
    if current.shape[0] == 0 or current.shape[1] == 0:
        raise ValueError("loss requires a nonempty current block")
    t = _sample_times(current.shape[0], current.device, config, generator)
    sigma, _ = loglinear_schedule(t, config.noise_eps)
    mask = torch.rand(current.shape, device=current.device, generator=generator) < ((1 - config.noise_eps) * t[:, None])
    noisy = torch.where(mask, k, current)
    past_input = past
    if model.training and not validation:
        if config.past_code_corruption_p > 0 and past.numel():
            corruption = torch.rand(past.shape, device=past.device, generator=generator) < config.past_code_corruption_p
            replacements = torch.randint(k, past.shape, device=past.device, generator=generator)
            past_input = torch.where(corruption, replacements, past)
        if config.past_label_smoothing > 0:
            past_input = F.one_hot(past_input, k + 1).float()
            uniform = torch.cat((torch.full((k,), 1/k, device=past.device), torch.zeros(1, device=past.device)))
            past_input = past_input * (1 - config.past_label_smoothing) + uniform * config.past_label_smoothing
    logits = model(noisy, past_input, sigma)
    # Cross-entropy on regular channels then select masked sites; no -inf * 0.
    nll = -F.log_softmax(logits[..., :k].float(), dim=-1).gather(-1, current[..., None]).squeeze(-1)
    return (torch.where(mask, nll, 0.) / t[:, None]).mean()


class EMA:
    """Named parameter EMA with portable, strict, non-aliasing checkpoint state."""
    def __init__(self, model, decay):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {name: value.detach().clone() for name, value in model.named_parameters()}

    def _validate_model(self, model):
        params = dict(model.named_parameters())
        if params.keys() != self.shadow.keys():
            raise ValueError("EMA parameter names do not match model")
        if any(value.shape != self.shadow[name].shape for name, value in params.items()):
            raise ValueError("EMA parameter shapes do not match model")
        return params

    @torch.no_grad()
    def update(self, model):
        for name, value in self._validate_model(model).items():
            shadow = self.shadow[name].to(device=value.device, dtype=value.dtype)
            shadow.lerp_(value.detach(), 1 - self.decay)
            self.shadow[name] = shadow
        self.num_updates += 1

    def state_dict(self):
        return {"decay": self.decay, "num_updates": self.num_updates,
                "shadow": {name: value.detach().clone() for name, value in self.shadow.items()}}

    def load_state_dict(self, state):
        if set(state) != {"decay", "num_updates", "shadow"}:
            raise ValueError("invalid EMA state fields")
        if not 0 <= state["decay"] < 1 or not isinstance(state["num_updates"], int) or state["num_updates"] < 0:
            raise ValueError("invalid EMA decay or update count")
        values = state["shadow"]
        if values.keys() != self.shadow.keys():
            raise ValueError("EMA state parameter names do not match")
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.shape != self.shadow[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid EMA state tensor: {name}")
        # Validate every entry before mutation; reject partially compatible states.
        self.shadow = {name: value.detach().to(device=self.shadow[name].device, dtype=self.shadow[name].dtype).clone()
                       for name, value in values.items()}
        self.decay, self.num_updates = float(state["decay"]), state["num_updates"]


@contextmanager
def inference_context(model, ema=None):
    """Exception-safe, nestable EMA/eval/no-grad scope preserving each module's mode.

    Each scope owns its backup rather than storing mutable backups in the EMA object.
    """
    modes = [(module, module.training) for module in model.modules()]
    parameters = None
    if ema is not None:
        parameters = ema._validate_model(model)
    live = {name: value.detach().clone() for name, value in parameters.items()} if parameters is not None else None
    try:
        if parameters is not None:
            with torch.no_grad():
                for name, value in parameters.items():
                    value.copy_(ema.shadow[name].to(device=value.device, dtype=value.dtype))
        model.eval()
        with torch.no_grad():
            yield model
    finally:
        if live is not None:
            with torch.no_grad():
                for name, value in parameters.items():
                    value.copy_(live[name])
        for module, training in modes:
            module.training = training


def _categorical(probabilities, generator):
    """Reference exponential-race categorical draw, using an explicit generator."""
    uniforms = torch.rand(probabilities.shape, device=probabilities.device, generator=generator)
    denominator = 1e-10 - torch.log(uniforms + 1e-10)
    return (probabilities / denominator).argmax(dim=-1)


def _sample_block(model, past, length, config, generator, stats, *, initial=None,
                  steps=None, start_time=1., nfe_key="generation_nfe"):
    k = model.config.codebook_size
    batch = past.shape[0]
    x = (torch.full((batch, length), k, device=past.device, dtype=torch.long)
         if initial is None else initial.clone())
    steps = config.steps if steps is None else steps
    timesteps = torch.linspace(start_time, min(config.sampling_eps, start_time / 2), steps + 1,
                               device=x.device, dtype=_time_dtype(config))
    p_cache = None
    for index in range(steps):
        t = timesteps[index].expand(batch)
        s = timesteps[index + 1]
        if p_cache is None:
            sigma, _ = loglinear_schedule(t, config.noise_eps)
            logits = model(x, past, sigma)
            p_cache = substitution_log_probs(logits, x, k, config.temperature).exp()
            stats[nfe_key] += 1
        # The factor 1-noise_eps cancels in the categorical normalization. Using
        # it here keeps cached and uncached DDPM exactly on the same schedule.
        move_t = (1 - config.noise_eps) * t[:, None, None]
        move_s = (1 - config.noise_eps) * s
        q = p_cache * (move_t - move_s)
        q[..., k] = move_s
        proposed = _categorical(q, generator)
        next_x = torch.where(x == k, proposed, x)
        reusable = config.predictor == "ddpm_cache" and not model.config.time_conditioning and torch.equal(next_x, x)
        if not reusable:
            p_cache = None
        x = next_x
    if torch.any(x == k):
        sigma, _ = loglinear_schedule(timesteps[-1].expand(batch), config.noise_eps)
        logits = model(x, past, sigma)
        stats[nfe_key] += 1
        # Terminal noise removal follows reference argmax while retaining anchors.
        x = torch.where(x == k, logits[..., :k].argmax(-1), x)
    return x


def _score_pool(batch, length, config, device, generator):
    # Deterministic midpoint strata avoid unstable t~0 scores. Fixed for the whole
    # candidate set, with separate per-document masks and a separate RNG stream.
    times = ((torch.arange(config.score_draws, device=device).float() + .5) / config.score_draws)
    times = times[:, None].expand(-1, batch)
    masks = torch.rand((config.score_draws, batch, length), device=device, generator=generator)
    masks = masks < ((1 - config.noise_eps) * times[..., None])
    if config.score_formula == "legacy_mask_count":
        # Historical generator forced position zero when an entire draw was empty.
        masks = masks.clone()
        for draw in range(config.score_draws):
            if not masks[draw].any():
                masks[draw, :, 0] = True
    return times, masks


def score_candidates(model, past, candidates, times, masks, config, stats=None):
    """Conditional diffusion score with a supplied, shared t/mask pool.

    candidates: Long[C,B,L]; times: Float[D,B]; masks: Bool[D,B,L].
    Returns Float[C,B]. Uses temperature 1 for scoring the trained distribution.
    Caller must hold inference_context so all calls use the same EMA/eval weights.
    """
    if candidates.ndim != 3 or candidates.dtype != torch.long:
        raise ValueError("candidates must be Long[C,B,L]")
    count, batch, length = candidates.shape
    if count == 0 or batch == 0 or length == 0:
        raise ValueError("candidates must have nonempty dimensions")
    if times.ndim != 2 or times.shape[1] != batch or times.shape[0] == 0 or masks.shape != (*times.shape, length) or masks.dtype != torch.bool:
        raise ValueError("score pool must be times[D,B] and masks[D,B,L]")
    if torch.any(times <= 0):
        raise ValueError("scoring times must be strictly positive")
    scores = torch.zeros(count, batch, device=candidates.device)
    k = model.config.codebook_size
    for candidate_index, candidate in enumerate(candidates):
        _validate_codes(candidate, k, name="candidate")
        for t, mask in zip(times, masks):
            sigma, _ = loglinear_schedule(t, config.noise_eps)
            noisy = torch.where(mask, k, candidate)
            logits = model(noisy, past, sigma)
            if stats is not None:
                stats["scoring_nfe"] += 1
            nll = -F.log_softmax(logits[..., :k].float(), dim=-1).gather(-1, candidate[..., None]).squeeze(-1)
            denom = (mask.sum(-1).clamp_min(1) if config.score_formula == "legacy_mask_count" else length)
            scores[candidate_index] += torch.where(mask, nll, 0.).sum(-1) / (t * denom)
    return scores / times.shape[0]


def _raw_probabilities(model, current, past, config, stats, nfe_key):
    sigma, _ = loglinear_schedule(torch.full((current.shape[0],), config.sampling_eps,
                                           device=current.device, dtype=_time_dtype(config)), config.noise_eps)
    logits = model(current, past, sigma)
    stats[nfe_key] += 1
    regular = F.softmax(logits[..., :model.config.codebook_size].float(), dim=-1)
    return F.pad(regular, (0, 1))


def _masked_position_confidence(model, current, past, config, stats):
    """Conditional P(original code | fixed past, all other current codes).

    Exactly one position per document is masked in each forward call. Revealed
    raw-logit outputs are not trained confidence estimates under mask-only loss.
    This leave-one-out heuristic costs current_length network calls per round.
    """
    k = model.config.codebook_size
    _validate_codes(current, k, name="refinement current")
    _validate_past(past, current, k)
    batch, length = current.shape
    if batch == 0 or length == 0:
        raise ValueError("refinement requires a nonempty current block")
    # One expected forward mask, subject to the model's trained time range.
    probe_time = max(config.sampling_eps, min(1., 1 / ((1 - config.noise_eps) * length)))
    t = torch.full((batch,), probe_time, device=current.device, dtype=_time_dtype(config))
    sigma, _ = loglinear_schedule(t, config.noise_eps)
    confidence = torch.empty((batch, length), device=current.device, dtype=torch.float32)
    for position in range(length):
        probe = current.clone()
        probe[:, position] = k
        logits = model(probe, past, sigma)
        stats["refinement_nfe"] += 1
        probabilities = F.softmax(logits[:, position, :k].float(), dim=-1)
        confidence[:, position] = probabilities.gather(-1, current[:, position:position + 1]).squeeze(-1)
    if not torch.isfinite(confidence).all():
        raise ValueError("refinement produced nonfinite masked conditional probabilities")
    return confidence


def _refine_block(model, current, past, config, generator, stats):
    for _ in range(config.refinement_rounds):
        confidence = _masked_position_confidence(model, current, past, config, stats)
        remask_count = max(1, math.ceil(current.shape[1] * config.refinement_fraction))
        indices = confidence.topk(remask_count, dim=-1, largest=False).indices
        initial = current.clone().scatter(-1, indices, model.config.codebook_size)
        current = _sample_block(model, past, current.shape[1], config, generator, stats,
                                initial=initial, steps=config.refinement_steps,
                                start_time=remask_count / current.shape[1], nfe_key="refinement_nfe")
    return current


def sample_codes(model, num_codes, batch_size, config: SamplingConfig, ema=None, seed=0, prefix=None):
    """Generate total num_codes INCLUDING prefix; keep supplied/previous past fixed.

    Full context is the default. max_context_codes=112 is an explicit legacy
    truncation ablation. Refinement remasks only the current block; it is not ReMDM.
    The two local RNG streams leave the caller's global torch RNG untouched.
    """
    _positive_int(num_codes, "num_codes")
    _positive_int(batch_size, "batch_size")
    device = next(model.parameters()).device
    k = model.config.codebook_size
    if prefix is None:
        codes = torch.empty((batch_size, 0), device=device, dtype=torch.long)
    else:
        _validate_codes(prefix, k, name="prefix")
        if prefix.shape[0] != batch_size or prefix.shape[1] > num_codes:
            raise ValueError("prefix batch must match batch_size and prefix length must not exceed num_codes")
        codes = prefix.to(device=device).clone()
    past_state = codes
    if config.past_mix_alpha < 1:
        past_state = F.one_hot(codes, k + 1).float()
    generation_rng = torch.Generator(device=device).manual_seed(int(seed))
    scoring_rng = torch.Generator(device=device).manual_seed((int(seed) + 0x5DEECE66D) % (2**63 - 1))
    stats = {"generation_nfe": 0, "scoring_nfe": 0, "refinement_nfe": 0,
             "max_prefix_codes": 0, "generated_blocks": 0,
              "refinement_algorithm": "experimental_current_block_masked_leave_one_out" if config.refinement_rounds else "none",
             "score_formula": config.score_formula}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with inference_context(model, ema):
        while codes.shape[1] < num_codes:
            length = min(model.config.block_length, num_codes - codes.shape[1])
            past = past_state if config.max_context_codes is None else past_state[:, -config.max_context_codes:]
            stats["max_prefix_codes"] = max(stats["max_prefix_codes"], past.shape[1])
            candidates = []
            # Entire candidate pool is generated before scoring; score draw count
            # cannot affect the generation stream or candidate ordering.
            for _ in range(config.candidates):
                candidate = _sample_block(model, past, length, config, generation_rng, stats)
                candidate = _refine_block(model, candidate, past, config, generation_rng, stats)
                candidates.append(candidate)
            stacked = torch.stack(candidates)
            if config.candidates > 1:
                times, masks = _score_pool(batch_size, length, config, device, scoring_rng)
                scores = score_candidates(model, past, stacked, times, masks, config, stats)
                selected = stacked[scores.argmin(0), torch.arange(batch_size, device=device)]
            else:
                selected = stacked[0]
            codes = torch.cat((codes, selected), dim=1)
            if config.past_mix_alpha < 1:
                soft = _raw_probabilities(model, selected, past, config, stats, "generation_nfe")
                hard = F.one_hot(selected, k + 1).float()
                representation = config.past_mix_alpha * hard + (1 - config.past_mix_alpha) * soft
            else:
                representation = selected
            past_state = torch.cat((past_state, representation), dim=1)
            stats["generated_blocks"] += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stats["wall_time_seconds"] = time.perf_counter() - started
    stats["total_nfe"] = stats["generation_nfe"] + stats["scoring_nfe"] + stats["refinement_nfe"]
    stats["returned_codes"] = codes.shape[1]
    return {"codes": codes, "stats": stats}
