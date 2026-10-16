"""Codec mathematical/state regression tests. Runtime deliberately not run at delivery."""
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from masked_longtext.codec import CodecConfig, TextCodec
from masked_longtext.vendor.codec.gpt_blocks import TokenProjection
from masked_longtext.vendor.codec.vq_vae_2 import Quantizer

torch.set_num_threads(1)


def tiny_config(**overrides):
    fields = dict(vocab_size=19, codebook_size=7, dim=16, num_heads=2,
                  num_layers=1, conv_dropout=0.0, dead_code_check_every=0,
                  logit_chunk_size=3)
    fields.update(overrides)
    return CodecConfig(**fields)


@pytest.mark.parametrize("bias", [False, True])
def test_integer_projection_preserves_dense_outputs_and_weight_gradients(bias):
    # Transposition mistakes or removal of a historical bias break this equality.
    projection = TokenProjection(5, 3, bias=bias)
    tokens = torch.tensor([[0, 4, 1], [1, 2, 0]])
    dense = F.one_hot(tokens, 5).float()
    actual = projection(tokens)
    expected = F.linear(dense, projection.weight, projection.bias)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    gradients = {n: p.grad.clone() for n, p in projection.named_parameters()}
    projection.zero_grad()
    expected.square().sum().backward()
    for name, parameter in projection.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name])
    # Soft distributions remain the original linear operation, including grads.
    soft = torch.rand(2, 3, 5, requires_grad=True)
    torch.testing.assert_close(projection(soft), F.linear(soft, projection.weight, projection.bias))
    projection(soft).sum().backward()
    assert soft.grad is not None
    assert projection.weight.shape == (3, 5)


def test_integer_projection_preserves_autocast_output_dtype():
    projection = TokenProjection(5, 3)
    tokens = torch.tensor([[0, 4, 1]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = projection(tokens)
        expected = F.linear(F.one_hot(tokens, 5).float(), projection.weight, projection.bias)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("length", [4, 12, 512])
def test_code_roundtrip_has_fourfold_compression_including_one_code(length):
    model = TextCodec(tiny_config()).eval()
    codes = model.encode_codes(torch.arange(length).remainder(19).reshape(1, length))
    assert codes.shape == (1, length // 4)
    assert codes.dtype == torch.long
    assert int(codes.min()) >= 0 and int(codes.max()) < 7
    logits = model.decode_codes(codes)
    assert logits.shape == (1, length, 19)
    assert torch.isfinite(logits).all()


def test_training_updates_each_quantizer_once_and_has_encoder_decoder_gradients():
    model = TextCodec(tiny_config()).train()
    output = model(torch.arange(24).remainder(19).reshape(2, 12))
    for name in ("loss", "ce", "embedding_loss", "commitment_loss", "top_reconstruction_loss"):
        assert output[name].shape == () and torch.isfinite(output[name])
    output["loss"].backward()
    assert int(model.model.top_quantizer.total_updates) == 1
    assert int(model.model.down_quantizer.total_updates) == 1
    assert model.model.encoders[0].transformer.wte.weight.grad.abs().sum() > 0
    assert model.model.decoders[1].final_layer.linear.weight.grad.abs().sum() > 0
    assert model.model.top_quantizer.code_embeddings.weight.grad is None


def test_inference_never_updates_codebook_and_restores_mixed_module_modes():
    model = TextCodec(tiny_config()).train()
    model.model.encoders[1].eval()
    modes = {name: module.training for name, module in model.named_modules()}
    before = {name: value.clone() for name, value in model.state_dict().items()}
    tokens = torch.arange(12).reshape(1, 12)
    codes = model.encode_codes(tokens)
    assert not model.decode_codes(codes).requires_grad
    assert not model.decode_token_ids(codes).requires_grad
    assert modes == {name: module.training for name, module in model.named_modules()}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_inference_restores_modes_after_real_downstream_exception():
    model = TextCodec(tiny_config()).train()
    model.model.encoders[1].eval()
    modes = [module.training for module in model.modules()]
    def fail(_module, _args):
        raise RuntimeError("intentional interruption")
    handle = model.model.encoders[0].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="intentional interruption"):
            model.encode_codes(torch.zeros(1, 4, dtype=torch.long))
    finally:
        handle.remove()
    assert modes == [module.training for module in model.modules()]


def test_quantizer_ema_uses_one_assignment_update_and_straight_through_gradient():
    quantizer = Quantizer(2, 2, use_ema=True, ema_decay=0.5, dead_code_check_every=0)
    with torch.no_grad():
        quantizer.code_embeddings.weight.copy_(torch.tensor([[0., 0.], [10., 10.]]))
        quantizer.ema_w.copy_(quantizer.code_embeddings.weight)
        quantizer.ema_cluster_size.fill_(1.)
        quantizer.total_updates.fill_(1)  # Already initialized; bypass first-batch seeding.
    z = torch.tensor([[[1., 1.], [9., 9.]]], requires_grad=True)
    result = quantizer.find_closest_embeddings(z)
    assert result.indices.flatten().tolist() == [0, 1]
    torch.testing.assert_close(quantizer.ema_w, torch.tensor([[0.5, 0.5], [9.5, 9.5]]))
    torch.testing.assert_close(quantizer.ema_cluster_size, torch.ones(2))
    assert int(quantizer.total_updates) == 2
    result.z_q.sum().backward()
    torch.testing.assert_close(z.grad, torch.ones_like(z))
    quantizer.eval()
    snapshot = {k: v.clone() for k, v in quantizer.state_dict().items()}
    quantizer.find_closest_embeddings(z.detach())
    for key, value in quantizer.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)


def test_chunked_checkpointed_head_matches_full_ce_and_gradients():
    full = TextCodec(tiny_config(use_ema=False, logit_chunk_size=0)).eval()
    chunked = TextCodec(tiny_config(use_ema=False, checkpoint_reconstruction=True)).eval()
    chunked.load_state_dict(full.state_dict(), strict=True)
    tokens = torch.arange(24).remainder(19).reshape(2, 12)
    full_result, chunked_result = full(tokens), chunked(tokens)
    torch.testing.assert_close(chunked_result["ce"], full_result["ce"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(chunked_result["loss"], full_result["loss"], rtol=1e-5, atol=1e-6)
    full_result["loss"].backward()
    chunked_result["loss"].backward()
    for (name, p), (_, q) in zip(full.named_parameters(), chunked.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("tokens", [torch.zeros(1, 5, dtype=torch.long),
    torch.empty(1, 0, dtype=torch.long), torch.zeros(0, 4, dtype=torch.long),
    torch.zeros(1, 4), torch.zeros(4, dtype=torch.long),
    torch.tensor([[-1, 0, 0, 0]]), torch.tensor([[19, 0, 0, 0]])])
def test_invalid_text_shape_dtype_or_range_is_rejected_before_updates(tokens):
    model = TextCodec(tiny_config())
    with pytest.raises((ValueError, TypeError)):
        model(tokens)
    assert int(model.model.top_quantizer.total_updates) == 0


@pytest.mark.parametrize("codes", [torch.zeros(1, 2), torch.tensor([[7]]),
    torch.tensor([[-1]]), torch.empty(1, 0, dtype=torch.long), torch.tensor([1, 2])])
def test_invalid_codes_including_mask_are_rejected(codes):
    with pytest.raises((ValueError, TypeError)):
        TextCodec(tiny_config()).decode_codes(codes)


@pytest.mark.parametrize("fields", [dict(dim=15), dict(num_heads=3), dict(num_layers=2),
    dict(codebook_size=0), dict(ema_decay=1.0), dict(conv_dropout=-0.1),
    dict(logit_chunk_size=-1), dict(beta=-1.0)])
def test_invalid_configuration_is_rejected(fields):
    with pytest.raises((ValueError, TypeError)):
        tiny_config(**fields)


def test_config_and_strict_state_roundtrip_retain_outputs():
    model = TextCodec(tiny_config()).eval()
    restored = TextCodec(CodecConfig(**asdict(model.config))).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    codes = torch.tensor([[0, 1]])
    torch.testing.assert_close(model.decode_codes(codes), restored.decode_codes(codes))


def test_wrapper_loss_matches_reference_vqvae_objective():
    wrapped = TextCodec(tiny_config(use_ema=False)).eval()
    reference = TextCodec(tiny_config(use_ema=False)).eval()
    reference.load_state_dict(wrapped.state_dict(), strict=True)
    tokens = torch.arange(12).reshape(1, 12)
    expected = reference.model(torch.ones(1), F.one_hot(tokens, 19).float(), tokens)
    result = wrapped(tokens)
    torch.testing.assert_close(result["ce"], expected.reconstructed_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(result["loss"], expected.loss, rtol=1e-5, atol=1e-6)
    enc = reference.model.encode(torch.ones(1), tokens)
    expected_codes = enc.min_encoding_indices_top.reshape(1, 3)
    assert torch.equal(wrapped.encode_codes(tokens), expected_codes)


@pytest.mark.parametrize("codes", [torch.tensor([[0]]), torch.tensor([[0, 1, 2], [2, 1, 0]])])
def test_chunked_token_decode_matches_full_joint_decode_argmax(codes):
    model = TextCodec(tiny_config(logit_chunk_size=3)).train()
    expected = model.decode_codes(codes).argmax(dim=-1)
    actual = model.decode_token_ids(codes)
    assert actual.shape == (codes.shape[0], codes.shape[1] * 4)
    assert actual.dtype == torch.long
    assert torch.equal(actual, expected)
    assert model.training
    assert int(model.model.top_quantizer.total_updates) == 0
    assert int(model.model.down_quantizer.total_updates) == 0
