# Static integration review: training, checkpoints, data and CLI

Reviewed source and written tests in `masked_longtext/training.py`, `checkpoint.py`, `data.py`, `__main__.py`, and `tests/test_{training,checkpoint,data}.py` against the codec and diffusion interfaces. This was a source review with AST parsing and compilation only. No imports of project/runtime dependencies, models, training, generation or pytest were executed during this review, in accordance with the user's explicit instruction.

## Material findings fixed

1. **Incomplete RNG resume.** NumPy was seeded but its RNG state was missing from checkpoints. `capture_rng` now serializes NumPy state as a tensor plus primitive fields; `restore_rng` restores it. This avoids ndarray pickle globals and remains compatible with `torch.load(weights_only=True)`. Python, CPU torch, per-device loss, CPU data and the selected CUDA dropout RNG state remain captured.

2. **Same tensor shapes could hide incompatible models.** Strict state-dict loading cannot detect a changed attention head count or disabled/enabled time conditioning. `restore_model` now checks dataclass model configuration before loading. Only explicit execution controls (`block_length`, `activation_checkpointing`, `logit_chunk_size`, `checkpoint_reconstruction`) are exempt. Exact resume separately requires the entire model configuration, including those controls, to match. Initialization with different architecture/objective construction settings is rejected instead of silently reinterpreting tensors. Checkpoint loading also rejects a reference source commit different from the selected commit.

3. **Document provenance file was never verified.** Prepared manifests contained `documents_sha256`, but loaders did not compare it to `documents.jsonl`. `read_manifest` now verifies the audit file when present, in addition to manifest identity. Shard arrays and row metadata checksums were already verified by `SequenceStore`.

4. **Validation omitted intermediate and long contexts.** The prior validation cycle previously used only zero, one block and the maximum; selecting a finite prefix of a full cycle would also omit the 2032-code maximum for 8192-token runs. `validation_contexts` now covers every block-aligned context plus a possible tail when the batch budget permits. A smaller budget uses a deterministic grid including both endpoints; a one-batch budget validates the maximum. Legacy 112-code validation therefore cycles through 0,16,...,112, while a budgeted 2048-code run still includes 2032. Past corruption remains disabled by `validation=True` and the inference scope.

5. **CUDA capability could be checked on the wrong GPU.** The bf16 capability query now runs under the explicitly selected CUDA device scope, including `cuda:1`, instead of querying an unrelated current device.

6. **Resume metadata and output could mislead.** Resume now requires complete optimizer/scaler/EMA/RNG/configuration/provenance fields and a valid integer optimizer step within the configured run. An existing `run_config.json` must describe the same run before it is overwritten/appended. Resuming an already completed checkpoint into a fresh output materializes `last.pt`; the function previously returned a path that did not exist because the training loop was skipped.

7. **Configuration validation allowed unusable numeric values.** Training counts now require actual integers; nonfinite LR/decay/clipping values and noninteger context limits are rejected before optimization.

## Interfaces and state review

- Prior training passes `Long[B,P]`, `Long[B,L]` and a device-matched explicit mask generator to `masked_loss`. Codebook sizes are checked against the code dataset. EMA is updated only after a successful optimizer step. Validation uses the exception-safe EMA/eval context and local validation generators.
- Resume loads AdamW state into the already constructed optimizer, preserving param groups and moment/step state. Learning rate is derived from the restored global optimizer step and unchanged full schedule configuration. Gradient accumulation is checkpointed only at completed optimizer boundaries; no partially accumulated gradient is claimed to be saved. fp16 scaler state is saved/restored. Skipped fp16 updates consume data/noise and codec forward state as usual; saved boundary state captures those effects, and a failed run resumes from its last saved boundary.
- `--init` creates a new optimizer/scaler/RNG run and, for the prior, copies the source EMA weights when available before starting a new EMA. Dataset/codec/tokenizer identities are checked; exact resume also compares package code hash and every training/loss/model field. Long text row length is a data/training setting, while prior block length remains independently configured.
- Source documents split by content identity before windows, exact duplicate documents are removed, windows remain contiguous and never cross documents, and unused tails are counted explicitly. Split membership is stable when document IDs/order change. Different prepared lengths preserve the same document split. Extraction preserves split/doc IDs and converts aligned token offsets to code offsets by four.
- CLI `_train`, `_encode`, `_sample` and `_load_codec` calls agree with the current core/codec interfaces and output shapes. Sampling enforces codec-file fingerprint and tokenizer/codebook identity before decoding. No CLI source was changed in this review.

## Written regression additions; not run

Added document-audit tamper detection and split invariance tests; same-shape attention-configuration rejection, permitted execution-control initialization, and wrong-source checkpoint tests; NumPy/Python RNG serialization round-trip, legacy/full-context validation endpoint, invalid numeric training configuration, and optimizer-boundary resume tests. The latter compares final model, AdamW state, EMA, scaler, RNG, step and best metric between uninterrupted and intentionally interrupted/resumed tiny prior runs, and checks an already-completed resume copy. All are **NOT RUN** under the current no-execution instruction.

## Remaining risks and limits

- Runtime/GPU/autocast behavior and optimizer-state restoration have not been exercised. Bitwise CUDA determinism is not established; exact resume requires the same device type and compatible hardware/runtime, and PyTorch backend nondeterminism remains possible.
- Validation uses a fixed finite batch/mask sample and budgeted context grid; it is not exhaustive corpus evaluation and cannot establish language quality. No text quality or speed gain is claimed.
- CLI model construction before loading checkpoints advances the caller's global torch RNG. Core sampling itself preserves global RNG and uses explicit seeds, so generated candidate streams are reproducible for fixed batching/configuration; the model-loading side effect is a minor library portability concern reported to the root owner.
- Dataset deduplication removes exact tokenized documents, not near duplicates or all repeated windows across different documents. There is no unrequested heuristic filtering or cross-document packing.
- Changed execution controls may be used for initialization, but architecture/objective construction differences now require an explicit audited migration. Historical Lightning files and missing/LFS-pointer weights remain unsupported.
- Failed preparation/training may leave partial output files; consumer manifests/checkpoints are published atomically, and fresh-output checks prevent silently reusing partial preparation. Checkpoint resume is intentionally restricted to optimizer boundaries.

Static AST/compile checks passed for the seven reviewed Python files. No runtime success is inferred from this result.
