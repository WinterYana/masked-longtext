"""Explicit portable checkpoint schema and strict loading."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from . import SOURCE_COMMIT, __version__


def code_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "package_version": __version__, "source_commit": SOURCE_COMMIT,
               "code_hash": code_hash(), **state}
    if payload.get("kind") not in ("codec", "prior") or "model" not in payload or "model_config" not in payload:
        raise ValueError("Checkpoint needs kind, model_config and model state")
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, *, expected_kind=None):
    path = Path(path)
    with path.open("rb") as handle:
        if handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Git LFS pointer, not actual weights: {path}")
    # No arbitrary unpickling of code/classes from a checkpoint.
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint. Historical Lightning files need an explicit audited conversion.")
    if data.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Checkpoint source commit differs from the selected reference")
    if data.get("kind") not in ("codec", "prior") or (expected_kind and data["kind"] != expected_kind):
        raise ValueError("Checkpoint kind mismatch")
    if not isinstance(data.get("model_config"), dict) or not isinstance(data.get("model"), dict):
        raise ValueError("Missing model configuration/state")
    return data


def restore_model(model, checkpoint):
    config = getattr(model, "config", None)
    if is_dataclass(config):
        # These settings change execution cost or the generated block size without
        # changing the interpretation of trained parameters. Attention heads and
        # time conditioning can differ despite identical state-dict shapes.
        execution_controls = {"block_length", "activation_checkpointing", "logit_chunk_size", "checkpoint_reconstruction"}
        actual = {k: v for k, v in asdict(config).items() if k not in execution_controls}
        saved = {k: v for k, v in checkpoint["model_config"].items() if k not in execution_controls}
        if actual != saved:
            raise ValueError("Checkpoint model configuration differs from the requested model architecture")
    return model.load_state_dict(checkpoint["model"], strict=True)
