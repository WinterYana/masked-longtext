import torch
import pytest

from masked_longtext.checkpoint import load_checkpoint, save_checkpoint, restore_model


def test_lfs_pointer_is_not_a_checkpoint(tmp_path):
    path = tmp_path / "best.pt"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
    with pytest.raises(ValueError, match="LFS"):
        load_checkpoint(path)


def test_strict_model_and_kind_loading(tmp_path):
    source = torch.nn.Linear(3, 2)
    path = tmp_path / "last.pt"
    save_checkpoint(path, {"kind": "prior", "model_config": {"test": True}, "model": source.state_dict(), "step": 3})
    data = load_checkpoint(path, expected_kind="prior")
    target = torch.nn.Linear(3, 2)
    restore_model(target, data)
    assert torch.equal(source.weight, target.weight)
    with pytest.raises(ValueError, match="kind"):
        load_checkpoint(path, expected_kind="codec")
    with pytest.raises(RuntimeError):
        restore_model(torch.nn.Linear(4, 2), data)


def test_semantically_different_attention_heads_rejected_despite_matching_weights():
    from dataclasses import asdict
    from masked_longtext.core import DenoiserConfig, PrefixDenoiser
    source = PrefixDenoiser(DenoiserConfig(codebook_size=4, hidden_size=16, n_blocks=1, n_heads=2, cond_dim=4))
    target = PrefixDenoiser(DenoiserConfig(codebook_size=4, hidden_size=16, n_blocks=1, n_heads=4, cond_dim=4))
    saved = {"model_config": asdict(source.config), "model": source.state_dict()}
    with pytest.raises(ValueError, match="configuration"):
        restore_model(target, saved)


def test_execution_controls_can_change_for_initialization():
    from dataclasses import asdict
    from masked_longtext.core import DenoiserConfig, PrefixDenoiser
    source = PrefixDenoiser(DenoiserConfig(codebook_size=4, hidden_size=8, n_blocks=1, n_heads=1, cond_dim=4))
    target = PrefixDenoiser(DenoiserConfig(codebook_size=4, hidden_size=8, n_blocks=1, n_heads=1, cond_dim=4,
                                         block_length=8, activation_checkpointing=True))
    restore_model(target, {"model_config": asdict(source.config), "model": source.state_dict()})
    assert torch.equal(target.backbone.vocab_embed.embedding, source.backbone.vocab_embed.embedding)


def test_checkpoint_wrong_reference_source_is_rejected(tmp_path):
    source = torch.nn.Linear(3, 2)
    path = tmp_path / "different_source.pt"
    save_checkpoint(path, {"kind": "prior", "model_config": {}, "model": source.state_dict(),
                           "source_commit": "another-reference"})
    with pytest.raises(ValueError, match="source commit"):
        load_checkpoint(path)
