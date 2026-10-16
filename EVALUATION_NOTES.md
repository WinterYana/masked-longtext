# Evaluation and offline judge implementation

Source protocol: `../docs/research/2026-10-14/judge_protocol.md`, version `latent-prose-judge-v1`. Source findings: `../docs/research/2026-10-14/REPORT.md`. No reference files were modified, no datasets inspected/downloaded, and no judge API calls made.

## Public evaluation API

```python
score_token_sequences(model, sequences, context_window, stride, device="cpu") -> list[dict]
evaluate_texts(texts, model_ref, context_window=1024, stride=512,
               device="cpu", local_files_only=True, revision=None) -> dict
text_metrics(text) -> dict
```

Each token-sequence result contains `num_tokens`, `num_scored_tokens`, `nll_sum`, `ppl`, `ppl_overflow`, and `per_position_nll`. Position zero is null because no BOS is synthesized. Empty/single-token documents have zero targets and null PPL. All later tokens, including real EOS, are targets. Floating-point overflow produces null PPL with an explicit overflow flag; nonfinite losses raise an error. Negative IDs, float IDs, Boolean IDs and multidimensional token tensors are rejected.

Input tokenization never truncates and never adds special tokens. Window length is evaluator context, not a document-length cap. `0 < stride < context_window` is enforced. The first window scores positions 1 through its last target; later windows begin scoring at the previous window's end. A length-10 document with window 8/stride 4 has exactly 9 targets, with contexts `[0:8]` and `[4:10]`. Documents are processed independently without padding or packing, so neighboring document lengths cannot change contexts. The scorer requires the caller's model to be on the requested device and restores every individual module's training flag after successful scoring or an exception.

`evaluate_texts` returns `documents`, `aggregate`, and `provenance`. The aggregate PPL is `exp(total NLL / total scored targets)`; mean finite document PPL is reported separately with its denominator. Model/tokenizer references, requested revision, resolved commit hashes when available, versions, text hashes and window settings are recorded. A missing resolved hash remains null. Transformers is imported lazily only by this optional evaluator. Its default is local files only. No real evaluator weights were loaded in this implementation session.

Word metrics are casefolded Unicode words with internal apostrophes/hyphens. Distinct 1–4 and repeated n-gram fractions are calculated on words, not model tokens. Repeated paragraphs count excess exact stripped paragraph copies; case/content within paragraphs is preserved. Diversity ratios depend on length and do not establish semantic quality.

## Offline judge API and exchange

```python
prepare_judge_packets(records, output_dir, *, length_condition=None,
                      token_counter=None, context_budget=None,
                      response_reserve=1024, segment_chars=1000) -> dict
validate_judge_response(response, packet_identity) -> dict
collect_judge_results(packet_dir, responses_path, output_path=None) -> dict
```

Each preparation record is `{"pair_id": "stable ID", "A": "full original text", "B": "full original text", "prompt": null, "metadata": {}}`; prompt/metadata are optional. A continuation has a nonempty real source prompt. The caller matches paired generation files by IDs before preparation. Metadata can hold generation configuration, seeds, provenance and hidden system identity; metadata never enters judge messages. Fixed-budget length conditions are generator token counts supplied by the caller, not measured judge-token counts.

`packets.jsonl` contains only `packet_id` and chat `messages`. Each pair is exported twice in deterministic AB and BA order. Only this file is intended for transfer to a judge. `mapping.json` stays local and holds original pair IDs, full texts, hashes, exact segment offsets, display-to-original mappings, metadata, budget status and response-envelope guidance. Segment boundaries preserve every input Unicode character, including whitespace/CRLF; long paragraphs are partitioned at fixed character boundaries. Candidate segments in messages are rendered as JSON strings with side-specific IDs, explicitly described as data. Judge messages contain the research rubric and no metadata/system names/PPL. Generated prose itself is preserved verbatim and may of course contain arbitrary words.

A supplied `context_budget` requires a `token_counter(messages)` callable using the judge's local tokenizer/chat template, including chat overhead. Both full candidates, instructions, prompt and reserved response tokens must fit before any packet file is written. Nothing is silently truncated. Without a budget, `token_budget_status` is explicitly `unchecked`; this does not establish fitness for a particular judge. Preparation performs no tokenization/network calls on its own.

External responses are saved as JSONL envelopes:

```json
{"packet_id":"ID from packets.jsonl","response":"raw JSON response or a decoded object","metadata":{"judge_model":"exact identifier","revision":"resolved revision","temperature":0,"input_tokens":0,"output_tokens":0,"cost":null}}
```

Metadata is optional because collection can process incomplete historical records. Record exact judge identity/revision, request parameters, times, token usage and costs when performing a real experiment. Collection retains metadata and the immutable raw response path/hash rather than calling a judge or overwriting its responses.

The response has exactly `coverage`, `A`, `B`, `winner`, `justification`, and `limitations`. Each side has exactly `scores`, `evidence`, and `unassessable`. All six rubric criteria must be present with integer 1–5 or null, rejecting floats and Booleans. Every null requires exactly one `{"criterion": ..., "reason": ...}` entry. Unconditional `prompt_adherence` must be null. Each of at most three evidence items has exactly `criterion`, `segments`, `quotes`, and `issue`. Segment/quote arrays are nonempty and have equal lengths; each quote must occur exactly in the corresponding same-side segment. Quotes are capped at 300 Unicode characters, a concrete short-quotation bound added to the protocol. This validates attribution, not whether an observation is scientifically correct. Duplicate JSON keys, nonfinite JSON constants and extra/missing fields are rejected.

Collection preserves all planned pairs. Original A/B wins require both orders to agree after reverse mapping. Two ties become a tie; other definite disagreements become `order_inconsistent`. Invalid, missing, incomplete and uncertain cases remain explicit and contribute to the denominator of all planned pairs. Unknown response IDs and malformed envelopes are retained and counted. Duplicate responses invalidate that packet rather than selecting a favorable retry; if using retries, resolve them externally under a prespecified policy and keep every raw response. Prompt/text/segment integrity is checked against preparation hashes. Criterion summaries describe all valid complete ordered replies, including order disagreements; these replies are not independent observations and are not confidence intervals.

## Verification status

Four Python sources (the two modules and their two test files) were parsed successfully with `ast.parse`. No production module was imported by this static check.

Before the user's later instruction to avoid execution arrived, the first test-first invocation was launched against the still-missing package. It reported 11 import-related failures and 13 judge temporary-directory setup errors; no pipeline training, generation, evaluator forward pass or judge call ran. Cancellation was requested immediately on receipt of the instruction; the invocation had already completed. This initial invocation is not a successful regression validation.

**Post-implementation tests were NOT RUN**, and the project suite was NOT RUN, following the user's instruction. Written tests cover full-target/final-window coverage, long-input suffix sensitivity, per-document independence, mixed-mode and exception restoration, strict window/ID validation, real EOS targets, nonfinite loss handling, token-weighted aggregation, lazy loader provenance, repetition counts, full-text judge blinding/swapping, source offsets, budget rejection, malformed/duplicate responses, evidence validation, all null reasons, ties, uncertain/incomplete outcomes, missing/unknown IDs, prompt tampering and raw-file protection. They are prepared checks, not claims of a passing runtime. Run them only if the user later authorizes execution; use a project-local pytest temporary directory where required by filesystem permissions.

This code is an offline evaluation framework. Natural-language generation quality, judge validity, GPU behavior, learned model quality and improvements over historical results remain unmeasured.
