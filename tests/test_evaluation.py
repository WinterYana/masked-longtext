import importlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

torch.set_num_threads(1)


class SyntheticLM(torch.nn.Module):
    """Position- and context-sensitive logits catch shifted/duplicated windows."""
    def __init__(self, fail=False):
        super().__init__()
        self.child = torch.nn.Dropout(0.5)
        self.fail = fail
        self.windows = []

    def forward(self, input_ids, attention_mask):
        assert not self.training and not self.child.training
        assert not torch.is_grad_enabled()
        assert attention_mask.eq(1).all()
        self.windows.append(input_ids[0].tolist())
        if self.fail:
            raise RuntimeError("synthetic failure")
        logits = torch.arange(32, dtype=torch.float64).view(1, 1, -1)
        scale = (input_ids.cumsum(1) + 1).unsqueeze(-1) / 30
        return SimpleNamespace(logits=logits * scale)


def api():
    return importlib.import_module("masked_longtext.evaluation")


def test_full_coverage_partial_window_and_padding_independence():
    model = SyntheticLM()
    model.train()
    model.child.eval()
    sequence = list(range(10))
    result = api().score_token_sequences(model, [sequence], 8, 4)[0]
    assert result["num_scored_tokens"] == 9
    assert result["num_tokens"] == 10
    assert result["per_position_nll"][0] is None
    assert len(result["per_position_nll"]) == 10
    assert model.windows == [list(range(8)), list(range(4, 10))]
    # Last positions use the last window; positions 6/7 must not be rescored.
    expected = []
    for position in range(1, 10):
        start = 0 if position < 8 else 4
        context_sum = sum(sequence[start:position]) + 1
        values = torch.arange(32, dtype=torch.float64) * context_sum / 30
        expected.append(float(torch.logsumexp(values, 0) - values[position]))
    assert result["per_position_nll"][1:] == pytest.approx(expected)
    assert result["nll_sum"] == pytest.approx(sum(expected))
    assert result["ppl"] == pytest.approx(math.exp(sum(expected) / 9))
    together = api().score_token_sequences(model, [sequence, list(range(12))], 8, 4)
    assert together[0] == result
    assert model.training and not model.child.training
    json.dumps(together, allow_nan=False)


def test_changes_beyond_first_window_change_score():
    model = SyntheticLM()
    results = api().score_token_sequences(model, [[1] * 20, [1] * 8 + [2] * 12], 8, 4)
    assert [r["num_scored_tokens"] for r in results] == [19, 19]
    assert results[0]["nll_sum"] != results[1]["nll_sum"]


@pytest.mark.parametrize("window,stride", [(8, 0), (8, 8), (8, 9), (1, 1), (8, -1), (8, 1.5)])
def test_rejects_windows_that_skip_causal_targets(window, stride):
    with pytest.raises(ValueError):
        api().score_token_sequences(SyntheticLM(), [[1, 2]], window, stride)


def test_empty_singleton_and_exception_restore_mixed_modes():
    model = SyntheticLM()
    model.train()
    model.child.eval()
    result = api().score_token_sequences(model, [[], [3]], 8, 4)
    assert [r["num_scored_tokens"] for r in result] == [0, 0]
    assert [r["ppl"] for r in result] == [None, None]
    assert result[1]["per_position_nll"] == [None]
    assert not model.windows
    model.fail = True
    with pytest.raises(RuntimeError, match="synthetic failure"):
        api().score_token_sequences(model, [[1, 2]], 8, 4)
    assert model.training and not model.child.training


def test_repetition_metrics_measure_words_and_paragraphs():
    result = api().text_metrics("one two one two\n\none two one two")
    assert result["num_words"] == 8
    assert result["num_paragraphs"] == 2
    assert result["distinct_1"] == pytest.approx(2 / 8)
    assert result["distinct_2"] == pytest.approx(2 / 7)
    assert result["repeated_paragraph_fraction"] == 0.5
    assert api().text_metrics("")["distinct_1"] == 0.0


def test_optional_loader_no_truncation_and_token_weighted_aggregate(monkeypatch):
    import sys
    class Tokenizer:
        name_or_path = "local-tokenizer"
        init_kwargs = {"_commit_hash": "tok-sha"}
        def __call__(self, text, **kwargs):
            assert kwargs["truncation"] is False
            assert kwargs["add_special_tokens"] is False
            return {"input_ids": [int(x) for x in text.split()]}
    class Loader:
        def __init__(self, value):
            self.value = value
        def from_pretrained(self, ref, **kwargs):
            assert kwargs["local_files_only"] is True
            assert kwargs["revision"] == "pinned"
            return self.value
    model = SyntheticLM()
    model.config = SimpleNamespace(_commit_hash="model-sha", max_position_embeddings=8)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=Loader(Tokenizer()), AutoModelForCausalLM=Loader(model), __version__="synthetic"))
    result = api().evaluate_texts(["0 1 2 3 4 5 6 7 8 9", "1 2"], "local", 8, 4, revision="pinned")
    assert result["aggregate"]["num_scored_tokens"] == 10
    expected = sum(d["nll_sum"] for d in result["documents"])
    assert result["aggregate"]["ppl"] == pytest.approx(math.exp(expected / 10))
    assert result["provenance"]["model_revision"] == "model-sha"
    assert result["provenance"]["tokenizer_revision"] == "tok-sha"
    assert result["provenance"]["truncation"] is False


@pytest.mark.parametrize("sequence", [[1, -1], [True, 2], [1.0, 2], torch.tensor([[1, 2]])])
def test_invalid_tokens_cannot_be_treated_as_text_or_padding(sequence):
    with pytest.raises(ValueError):
        api().score_token_sequences(SyntheticLM(), [sequence], 8, 4)


def test_tensor_logits_and_real_eos_remain_scored():
    class TensorLM(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return torch.zeros((*input_ids.shape, 4))
    result = api().score_token_sequences(TensorLM(), [[1, 0, 0]], 8, 4)[0]
    assert result["num_scored_tokens"] == 2
    assert result["ppl"] == pytest.approx(4.0)


def test_nonfinite_losses_raise_and_restore_original_mode():
    class BrokenLM(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return torch.full((*input_ids.shape, 4), float("nan"))
    model = BrokenLM()
    model.train()
    with pytest.raises(ValueError, match="nonfinite"):
        api().score_token_sequences(model, [[1, 2]], 8, 4)
    assert model.training
