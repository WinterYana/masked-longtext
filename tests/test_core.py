"""Regression tests for the prefix DiT process, state handling and RNG isolation."""
import importlib
import importlib.util
import math

import pytest
import torch
from torch import nn


@pytest.fixture
def core():
    assert importlib.util.find_spec("masked_longtext.core") is not None, "masked core is not implemented"
    return importlib.import_module("masked_longtext.core")


def small(core, **kw):
    fields = dict(codebook_size=4, hidden_size=16, n_blocks=2, n_heads=2,
                  cond_dim=8, dropout=0.1, block_length=4)
    fields.update(kw)
    return core.PrefixDenoiser(core.DenoiserConfig(**fields))


class ConstantDenoiser(nn.Module):
    """Known categorical distribution for independent sampler/loss expectations."""
    def __init__(self, config, logits=None):
        super().__init__()
        self.config = config
        self.logits = nn.Parameter(torch.tensor(logits or [0., 0., 0., 0., 99.]))
        self.dropout = nn.Dropout(.5)
        self.observations = []

    def forward(self, current, past, sigma):
        self.observations.append((current.clone(), past.clone(), sigma.clone(),
                                  self.training, self.logits.detach().clone()))
        return self.logits.expand(*current.shape, -1)


def constant(core, **kw):
    return ConstantDenoiser(core.DenoiserConfig(codebook_size=4, block_length=4, **kw))


def test_dit_soft_integer_prefix_equivalence_and_real_gradient(core):
    model = small(core).eval()
    # Zero-initialized reference head and adaLN gates need activation to exercise attention.
    with torch.no_grad():
        model.backbone.output_layer.linear.weight.normal_(std=.1)
        for block in model.backbone.blocks:
            block.adaLN_modulation.bias.fill_(.2)
    current = torch.tensor([[4, 2, 4, 1]])
    past = torch.tensor([[2, 3, 1]])
    sigma = torch.tensor([.5])
    a = model(current, past, sigma)
    soft = torch.nn.functional.one_hot(past, 5).float().requires_grad_()
    b = model(current, soft, sigma)
    assert a.shape == (1, 4, 5)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    b.square().sum().backward()
    assert soft.grad.abs().sum() > 0
    assert model.backbone.blocks[0].attn_qkv.weight.grad.abs().sum() > 0


def test_dit_zero_head_and_time_disabled(core):
    model = small(core).eval()
    current = torch.tensor([[4, 1, 2, 4]])
    past = torch.empty(1, 0, dtype=torch.long)
    assert torch.count_nonzero(model(current, past, torch.tensor([.2]))) == 0
    with torch.no_grad():
        model.backbone.output_layer.linear.weight.normal_()
    torch.testing.assert_close(model(current, past, torch.tensor([.1])),
                               model(current, past, torch.tensor([7.])), rtol=0, atol=0)


@pytest.mark.parametrize("bad", [dict(hidden_size=15, n_heads=2), dict(n_heads=0),
                                 dict(codebook_size=0), dict(dropout=1.1)])
def test_invalid_dit_config_rejected(core, bad):
    with pytest.raises(ValueError):
        small(core, **bad)


def test_substitution_is_exact_carryover_without_mask_probability(core):
    x = torch.tensor([[4, 2]])
    logits = torch.tensor([[[0., math.log(2), 0., 0., 1000.], [1., 2., 3., 4., 1000.]]])
    logp = core.substitution_log_probs(logits, x, 4)
    torch.testing.assert_close(logp[0, 0, :4].exp(), torch.tensor([.2, .4, .2, .2]))
    assert logp[0, 0, 4].exp() == 0
    torch.testing.assert_close(logp[0, 1].exp(), torch.tensor([0., 0., 1., 0., 0.]))


def test_loglinear_schedule_has_correct_mask_probability_and_weight(core):
    t = torch.tensor([.1, .5, 1.], dtype=torch.float64)
    sigma, dsigma = core.loglinear_schedule(t, noise_eps=.001)
    torch.testing.assert_close(-torch.expm1(-sigma), torch.tensor([.0999, .4995, .999], dtype=t.dtype))
    torch.testing.assert_close(dsigma / torch.expm1(sigma), 1 / t)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tiny_noise_epsilon_retains_finite_endpoint_without_schedule_clipping(core, dtype):
    # The old 1-(1-eps)*t arithmetic rounded the survival probability to zero.
    sigma, derivative = core.loglinear_schedule(torch.tensor([0., 1.], dtype=dtype), noise_eps=1e-8)
    assert torch.isfinite(sigma).all() and torch.isfinite(derivative).all()
    torch.testing.assert_close(sigma, torch.tensor([0., math.log(1e8)], dtype=torch.float32))
    assert derivative[1].item() == pytest.approx(1e8 - 1, rel=1e-6)
    assert torch.exp(-sigma[1]).item() == pytest.approx(1e-8, rel=1e-6)


def test_sub_float32_epsilon_promotes_schedule_and_sampling_times(core):
    sigma, derivative = core.loglinear_schedule(torch.tensor([1.]), noise_eps=1e-100)
    assert sigma.dtype == derivative.dtype == torch.float64
    assert torch.isfinite(sigma).all() and torch.isfinite(derivative).all()
    assert sigma.item() == pytest.approx(math.log(1e100))
    assert derivative.item() == pytest.approx(1e100)
    model = constant(core, time_conditioning=True)
    config = core.SamplingConfig(steps=2, noise_eps=1e-100, sampling_eps=1e-100)
    result = core.sample_codes(model, 4, 1, config, seed=3)
    assert (result["codes"] < 4).all()
    assert model.observations[0][2].dtype == torch.float64
    assert model.observations[0][2].item() == pytest.approx(math.log(1e100))
    assert all(torch.isfinite(observation[2]).all() for observation in model.observations)
    times = core._sample_times(2, torch.device("cpu"), core.LossConfig(sampling_eps=1e-100),
                               torch.Generator().manual_seed(3))
    assert times.dtype == torch.float64 and (times > 0).all()


def test_small_positive_times_preserve_sigma_and_analytic_weight(core):
    # log(survival) alone rounds survival to one at tiny float32 times.
    times = torch.tensor([1e-10, 1e-8], dtype=torch.float32)
    sigma, derivative = core.loglinear_schedule(times, noise_eps=.001)
    assert (sigma > 0).all()
    torch.testing.assert_close(sigma, torch.tensor([9.99e-11, 9.99e-9]))
    torch.testing.assert_close(derivative / torch.expm1(sigma), torch.tensor([1e10, 1e8]))


@pytest.mark.parametrize("field", ["sampling_eps", "noise_eps"])
def test_unrepresentable_epsilon_rejected_by_loss_and_sampling_configs(core, field):
    with pytest.raises(ValueError, match=field):
        core.LossConfig(**{field: 1e-320})
    with pytest.raises(ValueError, match=field):
        core.SamplingConfig(**{field: 1e-320})


def test_loss_weighted_mask_nll_divides_by_all_target_positions(core):
    model = constant(core)
    target = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 3]])
    cfg = core.LossConfig(past_code_corruption_p=0)
    gen = torch.Generator().manual_seed(4)
    loss = core.masked_loss(model, target[:, :0], target, cfg, generator=gen)
    # Independent integration estimate: -log p_target = log(4), weight = 1/t.
    x, _, sigma, _, _ = model.observations[-1]
    t = -torch.expm1(-sigma) / .999
    expected = math.log(4) * (((x == 4).sum(dim=1) / 4) / t).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(model.logits.grad).all()
    assert model.logits.grad[4] == 0


def test_no_mask_loss_is_zero_with_zero_gradient(core):
    model = constant(core)
    loss = core.masked_loss(model, torch.empty(1, 0, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long),
                            core.LossConfig(noise_eps=.999999, past_code_corruption_p=0),
                            generator=torch.Generator().manual_seed(0))
    assert loss == 0
    loss.backward()
    assert torch.count_nonzero(model.logits.grad) == 0


def test_past_corruption_regular_only_and_validation_clean(core):
    model = constant(core).train()
    past = torch.arange(4).repeat(64, 1)
    cur = past.clone()
    cfg = core.LossConfig(past_code_corruption_p=1, past_label_smoothing=.2)
    core.masked_loss(model, past, cur, cfg, generator=torch.Generator().manual_seed(1))
    corrupted = model.observations[-1][1]
    assert corrupted.shape == (64, 4, 5)
    assert torch.count_nonzero(corrupted[..., 4]) == 0
    assert not torch.equal(corrupted.argmax(-1), past)
    core.masked_loss(model, past, cur, cfg, generator=torch.Generator().manual_seed(1), validation=True)
    assert torch.equal(model.observations[-1][1], past)
    assert torch.equal(past, torch.arange(4).repeat(64, 1))


def test_ema_updates_and_roundtrips_without_aliasing(core):
    model = constant(core)
    ema = core.EMA(model, decay=.5)
    with torch.no_grad():
        model.logits.add_(2)
    ema.update(model)
    state = ema.state_dict()
    torch.testing.assert_close(state["shadow"]["logits"], torch.tensor([1., 1., 1., 1., 100.]))
    state["shadow"]["logits"].add_(100)
    assert ema.state_dict()["shadow"]["logits"][0] == 1
    other = core.EMA(model, .9)
    other.load_state_dict(ema.state_dict())
    assert other.decay == .5
    torch.testing.assert_close(other.state_dict()["shadow"]["logits"], ema.state_dict()["shadow"]["logits"])


def test_inference_exception_restores_exact_modes_and_parameters(core):
    model = constant(core).train()
    model.dropout.eval()
    ema = core.EMA(model, .5)
    with torch.no_grad():
        model.logits.add_(3)
    before = model.logits.detach().clone()
    with pytest.raises(RuntimeError, match="planned"):
        with core.inference_context(model, ema):
            assert not any(m.training for m in model.modules())
            assert model.logits[0] == 0
            raise RuntimeError("planned")
    torch.testing.assert_close(model.logits, before, rtol=0, atol=0)
    assert model.training and not model.dropout.training


def test_nested_inference_is_safe_and_restores_outer_state(core):
    model = constant(core).train()
    ema = core.EMA(model, .5)
    with torch.no_grad():
        model.logits.add_(3)
    with core.inference_context(model, ema):
        with core.inference_context(model, ema):
            assert model.logits[0] == 0
        assert model.logits[0] == 0 and not model.training
    assert model.logits[0] == 3 and model.training


def test_ema_schema_rejection_is_atomic(core):
    model = constant(core)
    ema = core.EMA(model, .5)
    state = ema.state_dict()
    state["shadow"]["logits"] = torch.zeros(6)
    with pytest.raises(ValueError):
        ema.load_state_dict(state)
    torch.testing.assert_close(ema.state_dict()["shadow"]["logits"], model.logits)
    assert ema.num_updates == 0


def test_all_sampling_phases_use_ema_and_restore_live_parameters(core):
    model = constant(core).train()
    ema = core.EMA(model, .5)
    with torch.no_grad():
        model.logits.add_(3)
    live = model.logits.detach().clone()
    core.sample_codes(model, 8, 1, core.SamplingConfig(steps=2, candidates=2, score_draws=2,
                                                      refinement_rounds=1, refinement_steps=2), ema=ema)
    assert model.observations and all(not obs[3] for obs in model.observations)
    assert all(obs[4][0] == 0 for obs in model.observations)
    torch.testing.assert_close(model.logits, live, rtol=0, atol=0)
    assert model.training


def test_time_conditioned_sampler_does_not_cache_changed_time(core):
    model = constant(core, time_conditioning=True)
    out = core.sample_codes(model, 4, 1, core.SamplingConfig(steps=16, predictor="ddpm_cache"), seed=3)
    assert out["stats"]["generation_nfe"] >= 16


def test_sampling_exception_restores_ema_and_mixed_modes(core):
    model = constant(core).train()
    model.dropout.eval()
    ema = core.EMA(model, .5)
    with torch.no_grad():
        model.logits.add_(3)
    live = model.logits.detach().clone()
    def fail(module, args):
        raise RuntimeError("planned sampler failure")
    hook = model.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="planned sampler failure"):
            core.sample_codes(model, 4, 1, core.SamplingConfig(steps=2), ema=ema)
    finally:
        hook.remove()
    torch.testing.assert_close(model.logits, live, rtol=0, atol=0)
    assert model.training and not model.dropout.training


@pytest.mark.parametrize("predictor", ["ddpm", "ddpm_cache"])
def test_sampler_final_codes_prefix_state_and_seed(core, predictor):
    model = constant(core).train()
    model.dropout.eval()
    prefix = torch.tensor([[1, 2], [2, 3]])
    cfg = core.SamplingConfig(steps=8, predictor=predictor, candidates=2, score_draws=3)
    rng_before = torch.get_rng_state().clone()
    first = core.sample_codes(model, 11, 2, cfg, seed=5, prefix=prefix)
    second = core.sample_codes(model, 11, 2, cfg, seed=5, prefix=prefix)
    assert torch.equal(first["codes"], second["codes"])
    assert first["codes"].shape == (2, 11)
    assert torch.equal(first["codes"][:, :2], prefix)
    assert ((first["codes"] >= 0) & (first["codes"] < 4)).all()
    assert model.training and not model.dropout.training
    assert all(not item[3] for item in model.observations)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert first["stats"]["total_nfe"] == sum(first["stats"][k] for k in ["generation_nfe", "scoring_nfe", "refinement_nfe"])


def test_ddpm_cache_matches_uncached_draws_and_reduces_nfe(core):
    model = constant(core)
    common = dict(steps=128, candidates=1)
    a = core.sample_codes(model, 4, 1, core.SamplingConfig(predictor="ddpm", **common), seed=7)
    b = core.sample_codes(model, 4, 1, core.SamplingConfig(predictor="ddpm_cache", **common), seed=7)
    assert torch.equal(a["codes"], b["codes"])
    assert b["stats"]["generation_nfe"] < a["stats"]["generation_nfe"]


def test_scoring_draws_do_not_change_generated_candidate_pool(core):
    # Constant logits => all candidates tie; selected first candidate reveals RNG changes.
    model = constant(core)
    a = core.sample_codes(model, 20, 1, core.SamplingConfig(steps=8, candidates=2, score_draws=1), seed=18)
    b = core.sample_codes(model, 20, 1, core.SamplingConfig(steps=8, candidates=2, score_draws=9), seed=18)
    assert torch.equal(a["codes"], b["codes"])


def test_full_context_2048_and_explicit_legacy_limit(core):
    model = constant(core)
    model.config = core.DenoiserConfig(codebook_size=4, block_length=16)
    result = core.sample_codes(model, 2048, 1, core.SamplingConfig(steps=1), seed=1)
    assert result["codes"].shape == (1, 2048)
    assert max(obs[1].shape[1] for obs in model.observations) == 2032
    model.observations.clear()
    result = core.sample_codes(model, 256, 1, core.SamplingConfig(steps=1, max_context_codes=112))
    assert max(obs[1].shape[1] for obs in model.observations) == 112


def test_real_dit_accepts_2032_prefix_codes(core):
    model = small(core, hidden_size=8, n_heads=1, n_blocks=1, cond_dim=4).eval()
    logits = model(torch.full((1, 16), 4), torch.zeros(1, 2032, dtype=torch.long), torch.ones(1))
    assert logits.shape == (1, 16, 5) and torch.isfinite(logits).all()


def test_shared_score_pool_per_target_length_and_named_legacy(core):
    model = constant(core)
    candidates = torch.tensor([[[0, 1, 2, 3]], [[3, 2, 1, 0]]])
    times = torch.tensor([[.5]])
    masks = torch.tensor([[[True, False, False, False]]])
    a = core.score_candidates(model, torch.empty(1, 0, dtype=torch.long), candidates, times, masks,
                               core.SamplingConfig(score_draws=1))
    b = core.score_candidates(model, torch.empty(1, 0, dtype=torch.long), candidates, times, masks,
                               core.SamplingConfig(score_draws=1, score_formula="legacy_mask_count"))
    torch.testing.assert_close(a, torch.full((2, 1), math.log(4)/2))
    torch.testing.assert_close(b, torch.full((2, 1), 2*math.log(4)))
    assert torch.equal(model.observations[0][0] == 4, model.observations[1][0] == 4)


def test_refinement_keeps_prefix_and_only_remasks_current_block(core):
    model = constant(core)
    prefix = torch.tensor([[3, 2, 1, 0]])
    out = core.sample_codes(model, 8, 1, core.SamplingConfig(steps=2, refinement_rounds=2,
                                                          refinement_fraction=.5, refinement_steps=2), prefix=prefix)
    assert torch.equal(out["codes"][:, :4], prefix)
    assert out["stats"]["refinement_nfe"] > 0
    assert (out["codes"] < 4).all()
    assert all(torch.equal(obs[1], prefix) for obs in model.observations)


def test_refinement_confidence_uses_each_masked_position_and_counts_every_probe(core):
    class ConditionalConfidence(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = core.DenoiserConfig(codebook_size=2, block_length=3)
            self.anchor = nn.Parameter(torch.zeros(()))
            self.inputs = []

        def forward(self, current, past, sigma):
            self.inputs.append((current.clone(), past.clone()))
            # At revealed sites the untrained outputs misleadingly prefer code0.
            logits = torch.tensor([10., 0., 1000.]).expand(*current.shape, -1).clone()
            # Genuine masked conditional probabilities for code0: .8, .2, .6.
            for position, p_zero in enumerate((.8, .2, .6)):
                mask = current[:, position] == 2
                logits[mask, position, 0] = math.log(p_zero)
                logits[mask, position, 1] = math.log(1 - p_zero)
            return logits + self.anchor * 0

    model = ConditionalConfidence()
    current = torch.tensor([[0, 0, 0], [0, 0, 0]])
    past = torch.tensor([[1, 0], [0, 1]])
    before_current, before_past = current.clone(), past.clone()
    config = core.SamplingConfig(refinement_rounds=1, refinement_fraction=1/3,
                                  refinement_steps=2, predictor="ddpm", sampling_eps=1e-5)
    stats = {"refinement_nfe": 0}
    with core.inference_context(model):
        confidence = core._masked_position_confidence(model, current, past, config, stats)
    torch.testing.assert_close(confidence, torch.tensor([[.8, .2, .6], [.8, .2, .6]]))
    assert stats["refinement_nfe"] == 3
    assert len(model.inputs) == 3
    for position, (observed, observed_past) in enumerate(model.inputs):
        expected = before_current.clone()
        expected[:, position] = 2
        assert torch.equal(observed, expected)
        assert torch.equal(observed_past, before_past)
    assert torch.equal(current, before_current) and torch.equal(past, before_past)
    model.inputs.clear()
    stats["refinement_nfe"] = 0
    with core.inference_context(model):
        refined = core._refine_block(model, current, past, config, torch.Generator().manual_seed(3), stats)
    # Position1 is least probable under its masked conditional; 0/2 are anchors.
    assert torch.equal(model.inputs[3][0], torch.tensor([[0, 2, 0], [0, 2, 0]]))
    assert torch.equal(refined[:, [0, 2]], before_current[:, [0, 2]])
    assert torch.equal(current, before_current) and torch.equal(past, before_past)
    assert all(torch.equal(observed_past, before_past) for _, observed_past in model.inputs)
    # Three confidence probes, two uncached DDPM steps, optional terminal cleanup.
    assert stats["refinement_nfe"] == len(model.inputs)
    assert stats["refinement_nfe"] in (5, 6)


@pytest.mark.parametrize("kw", [dict(steps=0), dict(candidates=0), dict(temperature=0),
                                dict(predictor="analytic"), dict(max_context_codes=0),
                                dict(refinement_fraction=2), dict(score_formula="unknown")])
def test_invalid_sampling_config_rejected(core, kw):
    with pytest.raises(ValueError):
        core.sample_codes(constant(core), 4, 1, core.SamplingConfig(**kw))
