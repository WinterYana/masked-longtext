"""Offline integration test, prepared for the training machine; not run locally."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np

from masked_longtext.codec import CodecConfig
from masked_longtext.core import DenoiserConfig, SamplingConfig
from masked_longtext.data import DatasetWriter, read_jsonl, read_manifest
from masked_longtext.training import train
from masked_longtext import __main__ as cli


def test_tiny_codec_to_prior_to_decoded_ids(tmp_path):
    text_data = tmp_path / "tokens"
    writer = DatasetWriter(text_data, kind="text_tokens", vocab_size=32, tokenizer_info={"name": "synthetic-test", "vocab_size": 32},
                           extra={"lengths": [64]})
    for split in ("train", "validation"):
        for i in range(3):
            writer.add(split, (np.arange(64) + i) % 32, doc_id=f"{split}-{i}", start=0)
    writer.finish()
    codec_dir = tmp_path / "codec"
    train_cfg = {"steps": 1, "microbatch": 1, "accumulation": 1, "warmup_steps": 0,
                 "validation_batches": 1, "precision": "fp32", "seed": 11}
    train("codec", data=text_data, row_length=64,
          model_config=asdict(CodecConfig(vocab_size=32, codebook_size=8, dim=16, num_heads=2, num_layers=1,
                                         conv_dropout=0., logit_chunk_size=8)),
          training_config=train_cfg, output=codec_dir, device="cpu")
    code_data = tmp_path / "codes"
    cli._encode(SimpleNamespace(data=text_data, codec=codec_dir / "best.pt", output=code_data,
                                batch_size=1, device="cpu", lengths=[64], shard_rows=2))
    assert read_manifest(code_data)["lengths"] == [16]
    prior_dir = tmp_path / "prior"
    train("prior", data=code_data, row_length=16,
          model_config=asdict(DenoiserConfig(codebook_size=8, hidden_size=32, n_blocks=1, n_heads=4,
                                            cond_dim=16, dropout=0., block_length=4)),
          training_config={**train_cfg, "max_context_codes": None}, output=prior_dir, device="cpu")
    sample_config = tmp_path / "sampling.json"
    sample_config.write_text(json.dumps(asdict(SamplingConfig(steps=4, candidates=2, score_draws=2))), encoding="utf-8")
    output = tmp_path / "samples"
    cli._sample(SimpleNamespace(checkpoint=prior_dir / "best.pt", codec=codec_dir / "best.pt",
                                config=sample_config, output=output, device="cpu", text_tokens=64,
                                num_samples=2, batch_size=1, seed=12, no_ema=False, tokenizer=None))
    rows = list(read_jsonl(output / "samples.jsonl"))
    assert len(rows) == 2
    assert all(len(row["input_ids"]) == 64 and len(row["codes"]) == 16 for row in rows)
    assert all(0 <= value < 8 for row in rows for value in row["codes"])
