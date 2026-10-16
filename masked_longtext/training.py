"""Single-device training with accumulation and complete optimizer-boundary resume."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .checkpoint import code_hash, load_checkpoint, restore_model, save_checkpoint
from .data import SequenceStore, write_json


@dataclass
class TrainingConfig:
    steps: int = 20000
    microbatch: int = 4
    accumulation: int = 16
    lr: float = 1e-4
    weight_decay: float = .01
    warmup_steps: int = 500
    min_lr_ratio: float = .1
    grad_clip: float = 1.0
    precision: str = "fp32"
    seed: int = 42
    validation_every: int = 500
    validation_batches: int = 16
    checkpoint_every: int = 500
    log_every: int = 20
    ema_decay: float = .999
    max_context_codes: int | None = 112
    context_policy: str = "legacy"
    context_zero_p: float = .1
    context_full_p: float = .55

    def __post_init__(self):
        for key in ("steps", "microbatch", "accumulation", "validation_every", "validation_batches", "checkpoint_every", "log_every"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.precision not in ("fp32", "bf16", "fp16") or self.context_policy not in ("legacy", "all"):
            raise ValueError("Unsupported precision or context policy")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if any(not math.isfinite(value) for value in (self.lr, self.weight_decay, self.grad_clip)) or self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= self.min_lr_ratio <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid LR/EMA ratio")
        if not (0 <= self.context_zero_p <= 1 and 0 <= self.context_full_p <= 1 and self.context_zero_p + self.context_full_p <= 1):
            raise ValueError("Invalid context probabilities")
        if self.max_context_codes is not None and (type(self.max_context_codes) is not int or self.max_context_codes < 0):
            raise ValueError("max_context_codes must be a nonnegative integer or None")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def learning_rate(step, config):
    if config.warmup_steps and step <= config.warmup_steps:
        return config.lr * step / config.warmup_steps
    fraction = min(1., max(0., (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)))
    return config.lr * (config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2)


def capture_rng(data_rng, loss_rng, device):
    numpy_state = np.random.get_state()
    state = {"torch_cpu": torch.get_rng_state(), "data": data_rng.get_state(), "loss": loss_rng.get_state(),
             "device_type": device.type, "python": random.getstate(),
             # Use a tensor and primitives: NumPy ndarray pickles are not accepted
             # by portable torch.load(weights_only=True).
             "numpy": {"bit_generator": numpy_state[0],
                       "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
                       "position": numpy_state[2], "has_gauss": numpy_state[3],
                       "cached_gaussian": numpy_state[4]}}
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, data_rng, loss_rng, device):
    if state["device_type"] != device.type:
        raise ValueError("Exact resume needs the same device type; use --init for a new optimization run")
    torch.set_rng_state(state["torch_cpu"])
    data_rng.set_state(state["data"])
    loss_rng.set_state(state["loss"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["bit_generator"], numpy_state["keys"].cpu().numpy().astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def choose_context(maximum, block_length, config, generator):
    maximum = int(maximum // block_length * block_length)
    if maximum <= 0:
        return 0
    middle = (list(range(block_length, maximum, block_length)) if config.context_policy == "all"
              else sorted({n for n in (block_length, 2 * block_length) if n < maximum}))
    choices, weights = [0, maximum], [config.context_zero_p, config.context_full_p]
    rest = 1 - sum(weights)
    if middle:
        choices += middle
        weights += [rest / len(middle)] * len(middle)
    else:
        weights[1] += rest
    if sum(weights) <= 0:
        return maximum
    selected = torch.multinomial(torch.tensor(weights), 1, generator=generator).item()
    return choices[selected]


def _batch(store, count, generator, device):
    indices = torch.randint(len(store), (count,), generator=generator).tolist()
    return torch.from_numpy(np.stack([store[i] for i in indices])).to(device)


def _prior_inputs(rows, block_length, config, generator, context=None):
    maximum = rows.shape[1] - block_length
    if maximum < 0:
        raise ValueError("Code row is shorter than one generated block")
    if config.max_context_codes is not None:
        maximum = min(maximum, config.max_context_codes)
    context = choose_context(maximum, block_length, config, generator) if context is None else min(context, maximum)
    starts = torch.randint(rows.shape[1] - context - block_length + 1, (len(rows),), generator=generator).tolist()
    past = torch.stack([row[start:start + context] for row, start in zip(rows, starts)])
    current = torch.stack([row[start + context:start + context + block_length] for row, start in zip(rows, starts)])
    return past, current


def validation_contexts(row_length, block_length, config, max_batches=None):
    """Cover the context cycle; if budget is smaller, retain both range endpoints."""
    maximum = row_length - block_length
    if maximum < 0:
        raise ValueError("Code row is shorter than one generated block")
    if config.max_context_codes is not None:
        maximum = min(maximum, config.max_context_codes)
    contexts = list(range(0, maximum + 1, block_length))
    if contexts[-1] != maximum:
        contexts.append(maximum)
    if max_batches is not None and len(contexts) > max_batches:
        if max_batches <= 0:
            raise ValueError("max_batches must be positive")
        if max_batches == 1:
            return [maximum]
        contexts = [contexts[round(index * (len(contexts) - 1) / (max_batches - 1))]
                    for index in range(max_batches)]
    return contexts


def _autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA; use fp32 on CPU")
    if precision == "bf16" and device.type == "cuda":
        # Capability queries use CUDA's current device, which may differ from a
        # requested cuda:1 on machines with heterogeneous GPUs.
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This CUDA device does not support bf16")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def train(kind, *, data, row_length, model_config, training_config, output, device="cuda", loss_config=None,
          resume=None, init=None):
    """Called explicitly on the training machine; nothing starts at import time."""
    from .codec import CodecConfig, TextCodec
    from .core import DenoiserConfig, PrefixDenoiser, EMA, LossConfig, inference_context, masked_loss

    if kind not in ("codec", "prior") or (resume and init):
        raise ValueError("Choose codec/prior and only one of --resume or --init")
    cfg = TrainingConfig(**training_config)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on this machine")
    train_store = SequenceStore(data, "train", row_length, expected_kind="text_tokens" if kind == "codec" else "vq_codes")
    val_store = SequenceStore(data, "validation", row_length, expected_kind=train_store.manifest["kind"])
    if not len(train_store) or not len(val_store):
        raise ValueError("Both train and validation need complete rows of the requested length")
    if int(row_length) % 4 and kind == "codec":
        raise ValueError("Codec text length must be divisible by four")
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Use a fresh output directory, or --resume for an existing run")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    model = TextCodec(CodecConfig(**model_config)) if kind == "codec" else PrefixDenoiser(DenoiserConfig(**model_config))
    expected_vocab = model.config.vocab_size if kind == "codec" else model.config.codebook_size
    if expected_vocab != train_store.manifest["vocab_size"]:
        raise ValueError("Data vocabulary and model vocabulary differ")
    model.to(device)
    criterion_config = LossConfig(**(loss_config or {})) if kind == "prior" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")
    ema = EMA(model, decay=cfg.ema_decay) if kind == "prior" else None
    data_rng = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    loss_rng = torch.Generator(device=device).manual_seed(cfg.seed + 2)
    manifest = train_store.manifest
    identity = {"dataset_id": manifest["dataset_id"], "row_length": int(row_length),
                "codec_id": manifest.get("codec_id"), "tokenizer": manifest["tokenizer"]}
    step, best_metric = 0, float("inf")
    if resume or init:
        saved = load_checkpoint(resume or init, expected_kind=kind)
        if kind == "prior" and saved.get("codec_id") != identity["codec_id"]:
            raise ValueError("Prior checkpoint uses a different codec; code identities are not interchangeable")
        if saved.get("tokenizer") != identity["tokenizer"]:
            raise ValueError("Checkpoint tokenizer differs from the dataset")
        restore_model(model, saved)
        if resume:
            required_resume = {"training_config", "loss_config", "optimizer", "scaler", "ema", "rng",
                               "step", "best_metric", "dataset_id", "row_length", "tokenizer"}
            if required_resume - saved.keys():
                raise ValueError(f"Checkpoint lacks complete resume state: {sorted(required_resume - saved.keys())}")
            if type(saved["step"]) is not int or not 0 <= saved["step"] <= cfg.steps:
                raise ValueError("Checkpoint optimizer step is outside the configured run")
            if not isinstance(saved["best_metric"], (int, float)) or math.isnan(saved["best_metric"]):
                raise ValueError("Checkpoint best metric is invalid")
            if saved["model_config"] != asdict(model.config) or saved["training_config"] != asdict(cfg):
                raise ValueError("Exact resume requires identical model and training settings; use --init for a changed experiment")
            if saved.get("loss_config") != (asdict(criterion_config) if criterion_config else None):
                raise ValueError("Resume loss configuration differs")
            if any(saved.get(key) != value for key, value in identity.items()):
                raise ValueError("Resume dataset/row length/codec identity mismatch")
            if saved.get("code_hash") != code_hash():
                raise ValueError("Code changed since checkpoint; audit the change and use --init for a new run")
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            if ema:
                ema.load_state_dict(saved["ema"])
            step, best_metric = saved["step"], saved["best_metric"]
            restore_rng(saved["rng"], data_rng, loss_rng, device)
        elif ema:
            # Start a new optimizer run from the selected checkpoint's evaluated EMA weights.
            if saved.get("ema") is not None:
                ema.load_state_dict(saved["ema"])
                with inference_context(model, ema):
                    initial_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(initial_weights, strict=True)
            ema = EMA(model, decay=cfg.ema_decay)

    run_info = {"kind": kind, "model": asdict(model.config), "training": asdict(cfg),
                "loss": asdict(criterion_config) if criterion_config else None,
                **identity, "device": str(device), "torch_version": str(torch.__version__),
                "code_hash": code_hash(), "resume_from": str(resume) if resume else None,
                "init_from": str(init) if init else None}
    existing_info_path = output / "run_config.json"
    if resume and existing_info_path.exists():
        previous_info = json.loads(existing_info_path.read_text(encoding="utf-8"))
        identity_fields = {"kind", "model", "training", "loss", "dataset_id", "row_length", "codec_id", "tokenizer"}
        if any(previous_info.get(key) != run_info.get(key) for key in identity_fields):
            raise ValueError("Output directory belongs to a different run; resume into an empty directory")
    write_json(output / "run_config.json", run_info)

    def objective(rows, batch_rng, mask_rng, *, validation=False, context=None):
        if kind == "codec":
            metrics = model(rows)
            return metrics["loss"], metrics["ce"]
        past, current = _prior_inputs(rows, model.config.block_length, cfg, batch_rng, context)
        value = masked_loss(model, past, current, criterion_config, generator=mask_rng, validation=validation)
        return value, value

    def validate():
        vrng = torch.Generator(device="cpu").manual_seed(cfg.seed + 10001)
        mrng = torch.Generator(device=device).manual_seed(cfg.seed + 10002)
        values = []
        # fork_rng preserves training dropout stream even if validation internals evolve.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), inference_context(model, ema), torch.no_grad():
            for index in range(cfg.validation_batches):
                rows = _batch(val_store, cfg.microbatch, vrng, device)
                context = None
                if kind == "prior":
                    contexts = validation_contexts(row_length, model.config.block_length, cfg, cfg.validation_batches)
                    context = contexts[index % len(contexts)]
                with _autocast(device, cfg.precision):
                    _, metric = objective(rows, vrng, mrng, validation=True, context=context)
                values.append(float(metric))
        return sum(values) / len(values)

    def checkpoint(path):
        save_checkpoint(path, {"kind": kind, "model_config": asdict(model.config), "model": model.state_dict(),
                              "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                              "ema": ema.state_dict() if ema else None, "training_config": asdict(cfg),
                              "loss_config": asdict(criterion_config) if criterion_config else None,
                              "rng": capture_rng(data_rng, loss_rng, device), "step": step,
                              "best_metric": best_metric, **identity})

    model.train()
    started, overflow_count = time.monotonic(), 0
    if step == cfg.steps:
        # Resume into a fresh output may already be complete. Materialize the
        # returned artifact path rather than reporting a nonexistent last.pt.
        checkpoint(output / "last.pt")
    while step < cfg.steps:
        optimizer.zero_grad(set_to_none=True)
        current_lr = learning_rate(step + 1, cfg)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        sum_loss = 0.
        for _ in range(cfg.accumulation):
            rows = _batch(train_store, cfg.microbatch, data_rng, device)
            with _autocast(device, cfg.precision):
                loss, _ = objective(rows, data_rng, loss_rng)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite training loss at optimizer step {step + 1}")
            sum_loss += float(loss.detach())
            scaler.scale(loss / cfg.accumulation).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if cfg.precision != "fp16" and not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("Non-finite gradient norm")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            overflow_count += 1
            if overflow_count >= 8:
                raise FloatingPointError("Repeated fp16 overflow; switch precision or reduce LR")
            continue
        overflow_count = 0
        step += 1
        if ema:
            ema.update(model)
        entry = {"step": step, "train_loss": sum_loss / cfg.accumulation, "lr": current_lr,
                 "grad_norm": float(grad_norm), "elapsed_seconds": time.monotonic() - started}
        if step % cfg.validation_every == 0 or step == cfg.steps:
            value = validate()
            if not math.isfinite(value):
                raise FloatingPointError("Non-finite validation metric")
            entry["validation_ce" if kind == "codec" else "validation_masked_loss"] = value
            if value < best_metric:
                best_metric = value
                checkpoint(output / "best.pt")
        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            checkpoint(output / "last.pt")
        if step % cfg.log_every == 0 or step == cfg.steps or "validation_ce" in entry or "validation_masked_loss" in entry:
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, allow_nan=False) + "\n")
            print(json.dumps(entry, allow_nan=False), flush=True)
    return {"step": step, "best_metric": best_metric, "last_checkpoint": str(output / "last.pt")}
