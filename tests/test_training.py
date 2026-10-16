from dataclasses import asdict

import pytest
import torch
import numpy as np
import random

from masked_longtext.training import (TrainingConfig, learning_rate, choose_context, capture_rng,
                                     restore_rng, validation_contexts)


def test_lr_and_context_boundaries():
    config = TrainingConfig(steps=10, warmup_steps=2, lr=.001)
    assert learning_rate(1, config) == pytest.approx(.0005)
    assert learning_rate(2, config) == pytest.approx(.001)
    assert learning_rate(10, config) == pytest.approx(config.lr * config.min_lr_ratio)
    rng = torch.Generator().manual_seed(11)
    draws = {choose_context(112, 16, config, rng) for _ in range(500)}
    assert draws <= {0, 16, 32, 112}
    config.context_policy = "all"
    draws = {choose_context(2032, 16, config, rng) for _ in range(1000)}
    assert 2032 in draws and max(draws) == 2032 and all(x % 16 == 0 for x in draws)


def test_rng_restore_preserves_future_draws():
    data_rng = torch.Generator().manual_seed(1)
    loss_rng = torch.Generator().manual_seed(2)
    state = capture_rng(data_rng, loss_rng, torch.device("cpu"))
    expected = (torch.rand(5), torch.rand(5, generator=data_rng), torch.rand(5, generator=loss_rng))
    restore_rng(state, data_rng, loss_rng, torch.device("cpu"))
    actual = (torch.rand(5), torch.rand(5, generator=data_rng), torch.rand(5, generator=loss_rng))
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def test_training_config_rejects_silent_precision_or_context_errors():
    with pytest.raises(ValueError):
        TrainingConfig(precision="fp8")
    with pytest.raises(ValueError):
        TrainingConfig(context_zero_p=.8, context_full_p=.8)
    assert asdict(TrainingConfig())["context_policy"] == "legacy"


def test_rng_checkpoint_weights_only_roundtrip_preserves_numpy_and_python(tmp_path):
    from masked_longtext.checkpoint import load_checkpoint, save_checkpoint
    data_rng = torch.Generator().manual_seed(1)
    loss_rng = torch.Generator().manual_seed(2)
    state = capture_rng(data_rng, loss_rng, torch.device("cpu"))
    path = tmp_path / "rng.pt"
    save_checkpoint(path, {"kind": "prior", "model_config": {}, "model": {}, "rng": state})
    expected_numpy = np.random.rand(6)
    expected_python = [random.random() for _ in range(6)]
    restore_rng(load_checkpoint(path)["rng"], data_rng, loss_rng, torch.device("cpu"))
    assert np.array_equal(np.random.rand(6), expected_numpy)
    assert [random.random() for _ in range(6)] == expected_python


def test_validation_contexts_cover_legacy_and_long_prefix_endpoints():
    legacy = TrainingConfig(max_context_codes=112)
    assert validation_contexts(128, 16, legacy) == [0, 16, 32, 48, 64, 80, 96, 112]
    long = TrainingConfig(max_context_codes=None)
    contexts = validation_contexts(2048, 16, long, max_batches=16)
    assert len(contexts) == 16 and contexts[0] == 0 and contexts[-1] == 2032
    assert len(set(contexts)) == len(contexts) and all(context % 16 == 0 for context in contexts)
    assert validation_contexts(2048, 16, long, max_batches=1) == [2032]


@pytest.mark.parametrize("fields", [dict(microbatch=1.5), dict(accumulation=True), dict(warmup_steps=.5),
                                    dict(lr=float("nan")), dict(max_context_codes=1.5)])
def test_training_config_rejects_noninteger_counts_and_nonfinite_optimizer_values(fields):
    with pytest.raises(ValueError):
        TrainingConfig(**fields)


def _assert_nested_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_state_equal(a, b)
    else:
        assert left == right


def test_prior_optimizer_boundary_resume_matches_uninterrupted_adamw(tmp_path, monkeypatch):
    """Written integration regression; run only on a subsequently authorized machine."""
    from masked_longtext.data import DatasetWriter
    from masked_longtext.checkpoint import load_checkpoint
    import masked_longtext.training as training
    data = tmp_path / "codes"
    writer = DatasetWriter(data, kind="vq_codes", vocab_size=4, tokenizer_info={"name": "test"},
                           extra={"codec_id": "synthetic-codec", "lengths": [8]})
    for split in ("train", "validation"):
        for index in range(8):
            writer.add(split, [(index + position) % 4 for position in range(8)], doc_id=f"{split}_{index}", start=0)
    writer.finish()
    model_config = dict(codebook_size=4, hidden_size=8, n_blocks=1, n_heads=1, cond_dim=4,
                        dropout=.1, block_length=2)
    config = dict(steps=3, microbatch=2, accumulation=2, warmup_steps=0, validation_every=2,
                  validation_batches=2, checkpoint_every=1, log_every=3, max_context_codes=6)
    common = dict(data=data, row_length=8, model_config=model_config, training_config=config,
                  device="cpu", loss_config={"past_code_corruption_p": .2})
    full = tmp_path / "full"
    training.train("prior", output=full, **common)
    partial = tmp_path / "partial"
    original_save = training.save_checkpoint
    def interrupt_after_saved_boundary(path, state):
        original_save(path, state)
        if state["step"] == 1 and path.name == "last.pt":
            raise RuntimeError("planned boundary interrupt")
    monkeypatch.setattr(training, "save_checkpoint", interrupt_after_saved_boundary)
    with pytest.raises(RuntimeError, match="planned boundary interrupt"):
        training.train("prior", output=partial, **common)
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    training.train("prior", output=partial, resume=partial / "last.pt", **common)
    expected, actual = load_checkpoint(full / "last.pt"), load_checkpoint(partial / "last.pt")
    for key in ("model", "optimizer", "ema", "scaler", "rng", "step", "best_metric"):
        _assert_nested_state_equal(expected[key], actual[key])
    completed_copy = tmp_path / "completed_copy"
    result = training.train("prior", output=completed_copy, resume=full / "last.pt", **common)
    assert (completed_copy / "last.pt").is_file()
    assert result["last_checkpoint"] == str(completed_copy / "last.pt")
    for key in ("model", "optimizer", "ema", "scaler", "rng", "step", "best_metric"):
        _assert_nested_state_equal(expected[key], load_checkpoint(completed_copy / "last.pt")[key])
