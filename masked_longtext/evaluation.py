"""Complete-document causal scoring and local text diagnostics; no dataset access."""
from __future__ import annotations

import hashlib
import math
import re
import sys
from numbers import Integral
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


def _validate_windows(context_window: int, stride: int) -> None:
    if (isinstance(context_window, bool) or not isinstance(context_window, Integral)
            or isinstance(stride, bool) or not isinstance(stride, Integral)
            or not 0 < stride < context_window):
        raise ValueError("Require integer 0 < stride < context_window for causal target coverage.")


def _perplexity(nll_sum: float, count: int) -> tuple[float | None, bool]:
    if not count:
        return None, False
    mean = nll_sum / count
    if mean > math.log(sys.float_info.max):
        return None, True
    return math.exp(mean), False


def _tokens(sequence: Sequence[int]) -> list[int]:
    if isinstance(sequence, torch.Tensor):
        if sequence.ndim != 1 or sequence.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise ValueError("Each sequence must be one-dimensional integer token IDs.")
        sequence = sequence.tolist()
    result = list(sequence)
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in result):
        raise ValueError("Token IDs must be nonnegative integers, without padding sentinels.")
    return [int(x) for x in result]


def score_token_sequences(
    model, sequences: Iterable[Sequence[int]], context_window: int, stride: int, device="cpu"
) -> list[dict]:
    """Score positions 1..N-1 once; position zero has no preceding context.

    Windows start at 0, stride, 2*stride, ... . The first window scores all its
    shifted targets, later windows only previously uncovered targets. Documents
    are processed independently, with all-one masks and no batching or padding.
    The caller places model parameters on ``device``. All original submodule
    training flags are restored, even when scoring raises an exception.
    """
    _validate_windows(context_window, stride)
    modes = [(module, module.training) for module in model.modules()]
    results = []
    try:
        model.eval()
        with torch.inference_mode():
            for sequence in sequences:
                ids = _tokens(sequence)
                length = len(ids)
                losses: list[float | None] = [None] * length
                covered_end = 1  # Position zero is explicitly unscorable.
                if length > 1:
                    for begin in range(0, length, int(stride)):
                        end = min(begin + int(context_window), length)
                        first_target = max(covered_end, begin + 1)
                        if first_target < end:
                            inputs = torch.tensor([ids[begin:end]], dtype=torch.long, device=device)
                            output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs))
                            logits = output if isinstance(output, torch.Tensor) else output.logits
                            if (logits.ndim != 3 or logits.shape[:2] != inputs.shape
                                    or logits.shape[-1] < 1):
                                raise ValueError("Evaluator must return logits shaped [1, window_length, vocabulary].")
                            offset = first_target - begin
                            target_logits = logits[0, offset - 1:end - begin - 1]
                            # Preserve float64 probes; upcast fp16/bf16 before loss arithmetic.
                            if target_logits.dtype in (torch.float16, torch.bfloat16):
                                target_logits = target_logits.float()
                            values = F.cross_entropy(target_logits, inputs[0, offset:], reduction="none")
                            values = values.detach().cpu().double().tolist()
                            if not all(math.isfinite(v) for v in values):
                                raise ValueError("Evaluator produced nonfinite token losses.")
                            losses[first_target:end] = values
                        covered_end = end
                        if end == length:
                            break
                if any(x is None for x in losses[1:]):
                    raise RuntimeError("Internal scoring error: uncovered next-token target.")
                nll_sum = math.fsum(x for x in losses if x is not None)
                count = max(0, length - 1)
                ppl, overflow = _perplexity(nll_sum, count)
                results.append({"ppl": ppl, "ppl_overflow": overflow, "nll_sum": nll_sum,
                                "num_scored_tokens": count, "num_tokens": length,
                                "per_position_nll": losses})
    finally:
        # Assign individually: recursive train() would destroy mixed child modes.
        for module, training in modes:
            module.training = training
    return results


def text_metrics(text: str) -> dict:
    """Whitespace-paragraph and casefolded Unicode word metrics, not LM tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    words = re.findall(r"\b\w+(?:['’\-]\w+)*\b", text.casefold(), flags=re.UNICODE)
    paragraphs = [p.strip() for p in re.split(r"\r?\n[ \t]*\r?\n", text) if p.strip()]
    result = {"num_characters": len(text), "num_words": len(words),
              "num_paragraphs": len(paragraphs),
              "repeated_paragraph_fraction": (
                  (len(paragraphs) - len(set(paragraphs))) / len(paragraphs) if paragraphs else 0.0)}
    for n in (1, 2, 3, 4):
        total = max(0, len(words) - n + 1)
        unique = len({tuple(words[i:i + n]) for i in range(total)})
        result[f"distinct_{n}"] = unique / total if total else 0.0
        result[f"repeated_{n}gram_fraction"] = (total - unique) / total if total else 0.0
    return result


def evaluate_texts(
    texts: Iterable[str], model_ref: str, context_window=1024, stride=512, device="cpu",
    local_files_only=True, revision=None,
) -> dict:
    """Load an optional causal LM/tokenizer and evaluate untruncated documents.

    No BOS/EOS is synthesized; real encoded EOS tokens are scored normally.
    Local loading is the default. Transformers is imported only here.
    """
    _validate_windows(context_window, stride)
    texts = list(texts)
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Every document must be a string.")
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Text PPL requires the optional transformers dependency and local evaluator files.") from exc
    kwargs = {"local_files_only": local_files_only, "revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_ref, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs).to(device)
    config = model.config
    limits = [getattr(config, field, None) for field in ("max_position_embeddings", "n_positions")]
    limits = [v for v in limits if isinstance(v, Integral) and v > 0]
    if limits and context_window > min(limits):
        raise ValueError(f"context_window={context_window} exceeds evaluator position capacity={min(limits)}.")
    sequences = [tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"] for text in texts]
    documents = score_token_sequences(model, sequences, context_window, stride, device)
    for doc, text in zip(documents, texts):
        doc["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc["text_metrics"] = text_metrics(text)
    nll = math.fsum(doc["nll_sum"] for doc in documents)
    count = sum(doc["num_scored_tokens"] for doc in documents)
    ppl, overflow = _perplexity(nll, count)
    document_ppls = [doc["ppl"] for doc in documents if doc["ppl"] is not None]
    aggregate = {"num_documents": len(documents), "num_scored_tokens": count, "nll_sum": nll,
                 "ppl": ppl, "ppl_overflow": overflow,
                 "mean_document_ppl": math.fsum(ppl / len(document_ppls) for ppl in document_ppls) if document_ppls else None,
                 "num_finite_document_ppls": len(document_ppls)}
    provenance = {
        "model_ref": str(model_ref), "requested_revision": revision,
        "model_revision": getattr(config, "_commit_hash", None),
        "tokenizer_ref": getattr(tokenizer, "name_or_path", str(model_ref)),
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "context_window": int(context_window), "stride": int(stride), "device": str(device),
        "local_files_only": bool(local_files_only), "truncation": False,
        "add_special_tokens": False, "scoring": "all next-token targets once, without cross-document context",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
    }
    return {"documents": documents, "aggregate": aggregate, "provenance": provenance}
