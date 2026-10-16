# Independent static review: prefix MDLM and CLI integration

Reviewed `masked_longtext/core.py`, `masked_longtext/vendor/dit.py`, related written tests, and evaluation/judge calls in `masked_longtext/__main__.py`. Reference source was read only through `git show` at commit `3ff606831e48be8069cb42a2afffa666051d53fa`, specifically the selected prefix-block diffusion, MDLM DiT, and MDLM time/parameterization functions. No other reference branch was consulted or modified.

**This review executed no tests, model imports, models, sampling, training, downloads, or judge calls.** Findings are source-derived. Line numbers describe the reviewed source before root-coordinated repairs.

## Material findings

**Repair status, 2026-10-14:** the three findings below were addressed in the delivered source. Judge budgeting uses the local judge's rendered chat template; declared and recorded lengths must agree and each export has one length. The schedule uses cancellation-free survival, precision promotion and an explicit supported lower epsilon bound. Additional refinement review replaced revealed-site raw logits with masked leave-one-out confidence and included all probes in NFE. Regression tests were written; they were not executed. The findings below preserve the source review record, not a claim that those defects remain.

### P1 — Judge CLI passes the wrong type to its token-budget counter

Location: `masked_longtext/__main__.py:244`; counterpart contract: `masked_longtext/judge.py`, `prepare_judge_packets`.

The CLI builds `lambda text: len(tokenizer.encode(text, add_special_tokens=False))`, but packet preparation calls `token_counter(messages)` with a list of role/content dictionaries. A judge export using `--judge-tokenizer` therefore passes chat dictionaries to a text-only encoder and fails rather than checking the context budget. Merely concatenating message content would also omit role/chat-template overhead and could incorrectly admit an over-budget request.

Suggested repair: count the actual local judge chat serialization, for example `len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))`. Require a valid local chat template or an explicitly configured accurate alternative; do not silently estimate or flatten. Include generation-prompt tokens and the separate requested response reserve. This affects a CLI integration path; the underlying preparation function's callable contract is consistent.

### P2 — Declared length condition can disagree with recorded generated lengths

Location: `masked_longtext/__main__.py:239` through the preparation call at line 245.

The CLI verifies only that A and B have equal `text_token_count` within a pair, then labels every packet using unchecked `args.text_tokens`. Two records generated at 512 text tokens can be exported with a purported 8192-token length condition; mixed-length pair collections receive one common label. This misstates a controlled factor of the experiment and could affect judging or later interpretation.

Suggested repair: validate positive integer counts when present, enforce a homogeneous known generation length for one export, and require any explicit `--text-tokens` to match it. Infer the common known length when the argument is absent, and retain actual original counts in local pair metadata. If lengths are genuinely unknown, keep the condition explicitly unspecified rather than asserting a measured count.

### P2 — Accepted small `noise_eps` can make the endpoint schedule infinite

Location: `masked_longtext/core.py:140`–150; accepted config ranges at `LossConfig.__post_init__` and `SamplingConfig.__post_init__`.

Both validators accept any `0 < noise_eps < 1`. The schedule computes `-log1p(-(1-noise_eps)*t)` and `(1-noise_eps)/(1-(1-noise_eps)*t)` in the tensor's dtype, with half/bfloat16 explicitly promoted only to float32. For a valid `noise_eps=1e-8` and float32 `t=1`, `1-noise_eps` rounds to one, causing both sigma and its derivative to become infinite. Sampling always starts at `t=1`, and `PrefixDenoiser.forward` rejects the infinite sigma. Thus this accepted parameter setting fails before denoising; the comment claiming a finite endpoint does not hold for the whole accepted range.

Suggested repair: compute the survival probability without cancellation, e.g. `(1-t) + noise_eps*t`, use an appropriately precise schedule calculation, or explicitly reject epsilon values below the representable supported bound. Preserve the stated log-linear schedule rather than silently clipping parameters. The shipped `noise_eps=1e-3` default is unaffected by this finding. This was not demonstrated by execution because execution is prohibited.

## Checked core behavior without a material defect found

- **Loss math and antithetic times:** the implementation matches the reference's stratified sampling formula. With `sigma=-log(1-(1-epsilon)t)`, the identity `sigma'/expm1(sigma)=1/t` is correct. The weighted masked NLL is averaged over all target positions, not just masked targets. Empty-mask draws contribute zero while remaining connected to the computation graph. Replacement and smoothing assign no MASK probability, and validation avoids past corruption.
- **SUBS and DDPM:** regular-channel normalization assigns MASK zero model probability, while revealed positions are carried forward exactly. The transition masses are proportional to `p(x0)*(move_t-move_s)` and `move_s` for MASK; the common log-linear factor cancels in their normalization. Cached and uncached paths share this transition. Reuse is permitted only for unchanged state with disabled time conditioning. Terminal cleanup touches only remaining MASK positions.
- **EMA and scope restoration:** each inference scope saves its own parameter copies and each submodule's individual training mode, rather than sharing an EMA backup slot. Validation happens before parameter mutation. The `finally` path restores live values and mixed modes, and nested scopes restore the enclosing state. Ordinary exceptions during copying or the scope body are covered; mutating parameter shapes inside the scope is outside this contract.
- **Scoring RNG isolation:** local generation/scoring generators are distinct. The complete candidate pool is generated before the shared mask/time pool is consumed. Changing scoring draws does not consume generation RNG. Different selected candidates can naturally affect subsequent conditional distributions; this is not RNG leakage.
- **Refinement anchors:** only the current generated block is remasked. Unselected revealed positions are retained by SUBS/DDPM and terminal cleanup, and supplied/previous prefix codes are never remasked. The confidence heuristic is correctly labeled an experimental current-block method rather than a validated ReMDM implementation.
- **Full past access:** the default passes the entire generated past, with an explicit optional suffix limit. An intended 2048-code sequence in 16-code blocks passes 2032 prefix codes to the last block. A supplied prefix and final short block are handled without padding or latent past compression.
- **DiT architecture:** integer and soft-prefix embeddings use the same matrix; head layout, Q/K-only rotary convention, adaptive normalization, residual gates, zero-initialized modulation/final projection, and bidirectional attention follow the selected reference. Projection of only the current slice is equivalent because the final layer is positionwise. The pure PyTorch SDPA path matches the reference fallback's dropout semantics; removed FlashAttention/Hub/config/JIT machinery and reconstructed settings are documented adaptations, not historical checkpoint compatibility.
- **Input validation:** regular/masked code ranges, tensor rank/dtype, soft-prefix normalization, matching batch/device, nonempty current blocks, sigma finiteness, head divisibility, and sampling dimensions/options are checked. The small-epsilon numerical hole above remains the material validation finding.

## Evaluation/judge integration compatibility

`_evaluate` correctly passes text documents and supported PPL options and consumes the `documents`/`aggregate` structure. Optional PPL loading remains local-only. `judge-collect` calls the correct `(packet_dir, responses_path, output_path)` interface. Pair matching rejects duplicate IDs, mismatched ID sets and differing real prompts. Blinded metadata is kept outside messages. The judge-token counting mismatch and unchecked length labels are the integration exceptions reported above.

Runtime regression tests remain **NOT RUN**. Static inspection cannot confirm GPU memory/throughput, stochastic behavior across hardware, learned natural-language quality, or a passing suite.
