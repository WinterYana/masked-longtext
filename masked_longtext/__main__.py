"""Explicit CLI entry point. Importing the package starts no work."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from .data import DatasetWriter, SequenceStore, canonical_hash, file_hash, prepare_jsonl, read_jsonl, read_manifest, write_json


def local_tokenizer(reference):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install the optional text dependencies: pip install -e '.[text]'") from error
    tokenizer = AutoTokenizer.from_pretrained(reference, local_files_only=True)
    vocabulary = tokenizer.get_vocab()
    identity = {"name": Path(reference).name if Path(reference).exists() else reference,
                "vocab_size": len(tokenizer), "vocabulary_hash": canonical_hash(vocabulary),
                "special_tokens": tokenizer.special_tokens_map,
                "eos_token_id": tokenizer.eos_token_id}
    if getattr(tokenizer, "is_fast", False):
        identity["tokenizer_hash"] = canonical_hash(tokenizer.backend_tokenizer.to_str())
    else:
        raise ValueError("Use a fast tokenizer so its complete normalization/tokenization pipeline can be fingerprinted")
    return tokenizer, identity


def _compatible_tokenizer(actual, expected):
    # Names/paths can change when a local tokenizer directory is moved.
    for key in ("vocab_size", "vocabulary_hash", "tokenizer_hash", "special_tokens", "eos_token_id"):
        if key in expected and expected[key] != actual.get(key):
            raise ValueError(f"Tokenizer differs from training: {key}")


def _config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    return value


def _prepare(args):
    tokenizer, identity = (local_tokenizer(args.tokenizer) if args.tokenizer else
                           (None, {"name": args.tokenizer_id, "vocab_size": args.vocab_size,
                                   "tokenizer_unverified": True}))
    if tokenizer is None and args.vocab_size is None:
        raise ValueError("Provide --tokenizer for text, or --vocab-size for pretokenized input_ids")
    if args.append_eos and (tokenizer is None or tokenizer.eos_token_id is None):
        raise ValueError("--append-eos requires a tokenizer with EOS")
    result = prepare_jsonl(args.input, args.output, lengths=args.lengths,
                          vocab_size=len(tokenizer) if tokenizer is not None else args.vocab_size,
                          tokenizer_info=identity, tokenizer=(lambda text: tokenizer.encode(text, add_special_tokens=False)) if tokenizer else None,
                          seed=args.seed, val_fraction=args.validation_fraction, test_fraction=args.test_fraction,
                          shard_rows=args.shard_rows, eos_id=tokenizer.eos_token_id if args.append_eos else None)
    return {"dataset_id": result["dataset_id"], "stats": result["stats"], "output": str(Path(args.output).resolve())}


def _train(args):
    from .training import train
    config = _config(args.config)
    allowed = {"model", "training", "loss", "text_length", "description"}
    if set(config) - allowed:
        raise ValueError(f"Unknown config keys: {set(config) - allowed}")
    kind = "codec" if args.command == "train-codec" else "prior"
    length = int(config["text_length"])
    if length < 4 or length % 4:
        raise ValueError("text_length must be divisible by four")
    return train(kind, data=args.data, row_length=length if kind == "codec" else length // 4,
                 model_config=config["model"], training_config=config["training"],
                 loss_config=config.get("loss"), output=args.output, device=args.device, resume=args.resume, init=args.init)


def _load_codec(path, device):
    from .checkpoint import load_checkpoint, restore_model
    from .codec import CodecConfig, TextCodec
    saved = load_checkpoint(path, expected_kind="codec")
    codec = TextCodec(CodecConfig(**saved["model_config"]))
    restore_model(codec, saved)
    return codec.to(device).eval(), saved


def _encode(args):
    import numpy as np
    import torch
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    manifest = read_manifest(args.data)
    if manifest["kind"] != "text_tokens":
        raise ValueError("encode requires text_tokens data")
    device = torch.device(args.device)
    codec, saved = _load_codec(args.codec, device)
    if saved.get("tokenizer") != manifest["tokenizer"] or codec.config.vocab_size != manifest["vocab_size"]:
        raise ValueError("Codec and token dataset identities differ")
    lengths = sorted(set(args.lengths or manifest["lengths"]))
    if any(length not in manifest["lengths"] for length in lengths):
        raise ValueError("Requested length was not prepared")
    writer = DatasetWriter(args.output, kind="vq_codes", vocab_size=codec.config.codebook_size,
                           tokenizer_info=manifest["tokenizer"], shard_rows=args.shard_rows,
                           extra={"source_dataset_id": manifest["dataset_id"], "codec_id": file_hash(args.codec),
                                  "codec_config": asdict(codec.config), "compression_factor": 4,
                                  "offset_unit": "codes", "lengths": [n // 4 for n in lengths],
                                  "text_lengths": lengths, "codec_source_code_hash": saved.get("code_hash")})
    count = 0
    with torch.no_grad():
        for split in ("train", "validation", "test"):
            for length in lengths:
                store = SequenceStore(args.data, split, length, expected_kind="text_tokens")
                pending = []

                def flush():
                    nonlocal count
                    if not pending:
                        return
                    batch = torch.from_numpy(np.stack([row for row, _ in pending])).to(device)
                    codes = codec.encode_codes(batch).cpu().numpy()
                    if codes.shape != (len(pending), length // 4):
                        raise RuntimeError("Codec compression shape mismatch")
                    for row, (_, meta) in zip(codes, pending):
                        writer.add(split, row, doc_id=meta["doc_id"], start=meta["start"] // 4)
                    count += len(pending)
                    pending.clear()

                for record in store.iter_records():
                    pending.append(record)
                    if len(pending) >= args.batch_size:
                        flush()
                flush()
    result = writer.finish({"encoded_rows": count})
    return {"dataset_id": result["dataset_id"], "codec_id": result["codec_id"], "encoded_rows": count}


def _sample(args):
    import torch
    from .checkpoint import load_checkpoint, restore_model
    from .core import DenoiserConfig, PrefixDenoiser, EMA, SamplingConfig, sample_codes
    if args.num_samples < 1 or args.batch_size < 1 or args.text_tokens < 4 or args.text_tokens % 4:
        raise ValueError("Positive sample/batch counts and text token length divisible by four are required")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Sample output directory must be empty")
    saved = load_checkpoint(args.checkpoint, expected_kind="prior")
    codec_digest = file_hash(args.codec)
    if saved.get("codec_id") != codec_digest:
        raise ValueError("Wrong codec for this prior checkpoint")
    device = torch.device(args.device)
    codec, codec_saved = _load_codec(args.codec, device)
    if saved.get("tokenizer") != codec_saved.get("tokenizer"):
        raise ValueError("Prior and codec tokenizer identities differ")
    model = PrefixDenoiser(DenoiserConfig(**saved["model_config"]))
    restore_model(model, saved)
    model.to(device).eval()
    if model.config.codebook_size != codec.config.codebook_size:
        raise ValueError("Prior and codec codebooks have different sizes")
    ema = None
    if not args.no_ema:
        if saved.get("ema") is None:
            raise ValueError("Checkpoint lacks EMA; choose --no-ema explicitly if intended")
        ema = EMA(model, decay=saved["ema"]["decay"])
        ema.load_state_dict(saved["ema"])
    config_data = _config(args.config)
    sampling = SamplingConfig(**config_data.get("sampling", config_data))
    tokenizer = None
    if args.tokenizer:
        tokenizer, identity = local_tokenizer(args.tokenizer)
        _compatible_tokenizer(identity, saved.get("tokenizer", {}))
    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started, batches = time.perf_counter(), []
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle, torch.no_grad():
        for offset in range(0, args.num_samples, args.batch_size):
            size = min(args.batch_size, args.num_samples - offset)
            result = sample_codes(model, args.text_tokens // 4, size, sampling, ema=ema, seed=args.seed + offset)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decode_start = time.perf_counter()
            ids = codec.decode_token_ids(result["codes"])
            if ids.shape != (size, args.text_tokens):
                raise RuntimeError("Decoded output has unexpected token length")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batches.append({**result["stats"], "batch_size": size, "decode_seconds": time.perf_counter() - decode_start})
            for within, (codes, tokens) in enumerate(zip(result["codes"].cpu().tolist(), ids.cpu().tolist())):
                record = {"id": offset + within + 1, "input_ids": tokens, "codes": codes,
                          "text_token_count": len(tokens), "generation_batch_seed": args.seed + offset}
                if tokenizer is not None:
                    record["text"] = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    duration = time.perf_counter() - started
    result = {"checkpoint_sha256": file_hash(args.checkpoint), "codec_id": codec_digest,
              "sampling": asdict(sampling), "ema": ema is not None, "seed": args.seed,
              "text_tokens": args.text_tokens, "num_samples": args.num_samples, "batch_size": args.batch_size,
              "device": str(device), "torch_version": str(torch.__version__), "batches": batches,
              "total_seconds_including_decode_and_output": duration,
              "text_tokens_per_second": args.num_samples * args.text_tokens / duration,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "text_decoded": tokenizer is not None}
    write_json(output / "generation.json", result)
    return result


def _reconstruct(args):
    """Inspect the codec ceiling on held-out rows before training a prior."""
    import torch
    if args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Reconstruction output directory must be empty")
    store = SequenceStore(args.data, args.split, args.text_tokens, expected_kind="text_tokens")
    if not len(store):
        raise ValueError("No complete rows for this split/length")
    codec, saved = _load_codec(args.codec, args.device)
    if saved.get("tokenizer") != store.manifest["tokenizer"]:
        raise ValueError("Codec and reconstruction data use different tokenizers")
    tokenizer = None
    if args.tokenizer:
        tokenizer, identity = local_tokenizer(args.tokenizer)
        _compatible_tokenizer(identity, store.manifest["tokenizer"])
    output.mkdir(parents=True, exist_ok=True)
    correct, total, count = 0, 0, 0
    usage = torch.zeros(codec.config.codebook_size, dtype=torch.long)
    losses = []
    with (output / "reconstructed.jsonl").open("w", encoding="utf-8") as recon_file, \
         (output / "originals.jsonl").open("w", encoding="utf-8") as source_file, torch.no_grad():
        for row, meta in store.iter_records():
            if count >= args.max_samples:
                break
            tokens = torch.from_numpy(row).unsqueeze(0).to(args.device)
            codes = codec.encode_codes(tokens)
            reconstructed = codec.decode_token_ids(codes)
            if reconstructed.shape != tokens.shape:
                raise RuntimeError("Codec reconstruction length mismatch")
            matched = int((reconstructed == tokens).sum())
            correct += matched
            total += tokens.numel()
            count += 1
            usage += torch.bincount(codes.flatten().cpu(), minlength=len(usage))
            original_ids, reconstructed_ids = row.tolist(), reconstructed[0].cpu().tolist()
            common = {"id": count, "doc_id": meta["doc_id"], "start": meta["start"], "text_token_count": len(row)}
            original = {**common, "input_ids": original_ids}
            record = {**common, "input_ids": reconstructed_ids, "target_ids": original_ids,
                      "codes": codes[0].cpu().tolist(), "token_accuracy": matched / len(row)}
            if args.with_loss:
                metrics = codec(tokens)
                record["codec_losses"] = {key: float(value) for key, value in metrics.items()}
                losses.append(record["codec_losses"]["ce"])
            if tokenizer is not None:
                original["text"] = tokenizer.decode(original_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                record["text"] = tokenizer.decode(reconstructed_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                record["source_text"] = original["text"]
            source_file.write(json.dumps(original, ensure_ascii=False, allow_nan=False) + "\n")
            recon_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    result = {"codec_id": file_hash(args.codec), "dataset_id": store.manifest["dataset_id"],
              "split": args.split, "text_tokens": args.text_tokens, "num_samples": count,
              "selection": "first rows in manifest order; diagnostic, not an unbiased corpus estimate",
              "token_accuracy": correct / total, "mean_ce": sum(losses) / len(losses) if losses else None,
              "top_code_counts": usage.tolist(), "used_top_codes": int((usage > 0).sum()),
              "codebook_size": len(usage), "text_decoded": tokenizer is not None}
    write_json(output / "reconstruction.json", result)
    return result


def _evaluate(args):
    from .evaluation import evaluate_texts, text_metrics
    records = list(read_jsonl(args.input))
    if not records or any(not isinstance(r.get("text"), str) for r in records):
        raise ValueError("Evaluation needs nonempty records containing decoded text; sample with --tokenizer")
    texts = [record["text"] for record in records]
    result = evaluate_texts(texts, args.model, context_window=args.context_window, stride=args.stride,
                            device=args.device, local_files_only=True) if args.model else {
                                "documents": [{"text_metrics": text_metrics(text)} for text in texts],
                                "provenance": {"ppl": "not_requested"}}
    for row, record in zip(result["documents"], records):
        row["id"] = record.get("id")
    write_json(args.output, result)
    return {"output": str(Path(args.output).resolve()), "documents": len(records), "aggregate": result.get("aggregate")}


def _judge_prepare(args):
    from .judge import prepare_judge_packets
    first, second = {}, {}
    for path, target in ((args.a, first), (args.b, second)):
        for record in read_jsonl(path):
            if "id" not in record or not isinstance(record.get("text"), str):
                raise ValueError("Judge inputs require id and text")
            key = str(record["id"])
            if key in target:
                raise ValueError(f"Duplicate sample ID {key}")
            target[key] = record
    if first.keys() != second.keys() or not first:
        raise ValueError("Judge inputs must contain identical nonempty ID sets")
    pairs, observed_lengths = [], set()
    for key in first:
        if first[key].get("prompt") != second[key].get("prompt"):
            raise ValueError(f"Different prompts for pair {key}")
        counts = [first[key].get("text_token_count"), second[key].get("text_token_count")]
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError(f"Judge inputs need positive text_token_count for pair {key}")
        if counts[0] != counts[1]:
            raise ValueError(f"Different output lengths for pair {key}")
        if args.text_tokens is not None and counts[0] != args.text_tokens:
            raise ValueError(f"Pair {key} does not match the declared length condition")
        observed_lengths.add(counts[0])
        pairs.append({"pair_id": key, "A": first[key]["text"], "B": second[key]["text"],
                      "prompt": first[key].get("prompt"), "metadata": {"A_source": str(args.a), "B_source": str(args.b)}})
    if len(observed_lengths) != 1:
        raise ValueError("Export each text length as a separate judge comparison")
    tokenizer = local_tokenizer(args.judge_tokenizer)[0] if args.judge_tokenizer else None
    if tokenizer is not None and not getattr(tokenizer, "chat_template", None):
        raise ValueError("Judge budget checking requires the actual judge tokenizer with its chat template")
    counter = (lambda messages: len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))) if tokenizer else None
    return prepare_judge_packets(pairs, args.output, length_condition=next(iter(observed_lengths)),
                                 token_counter=counter, context_budget=args.context_budget, response_reserve=args.response_reserve)


def parser():
    root = argparse.ArgumentParser(description="Portable masked latent diffusion. Commands run only when explicitly invoked.")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare local JSONL documents; never downloads a corpus")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--tokenizer", help="Locally available fast tokenizer path/cache ID")
    prepare.add_argument("--tokenizer-id", default="provided-token-ids")
    prepare.add_argument("--vocab-size", type=int)
    prepare.add_argument("--lengths", nargs="+", type=int, default=[512, 8192])
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--validation-fraction", type=float, default=.01)
    prepare.add_argument("--test-fraction", type=float, default=.01)
    prepare.add_argument("--shard-rows", type=int, default=256)
    prepare.add_argument("--append-eos", action="store_true")
    prepare.set_defaults(handler=_prepare)
    for command in ("train-codec", "train-prior"):
        train = commands.add_parser(command)
        for name in ("data", "config", "output"):
            train.add_argument(f"--{name}", required=True)
        train.add_argument("--device", default="cuda")
        restart = train.add_mutually_exclusive_group()
        restart.add_argument("--resume")
        restart.add_argument("--init")
        train.set_defaults(handler=_train)
    encode = commands.add_parser("encode")
    for name in ("data", "codec", "output"):
        encode.add_argument(f"--{name}", required=True)
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--lengths", type=int, nargs="+")
    encode.add_argument("--batch-size", type=int, default=1)
    encode.add_argument("--shard-rows", type=int, default=256)
    encode.set_defaults(handler=_encode)
    reconstruct = commands.add_parser("reconstruct", help="Inspect held-out original to codec reconstruction")
    for name in ("data", "codec", "output"):
        reconstruct.add_argument(f"--{name}", required=True)
    reconstruct.add_argument("--tokenizer")
    reconstruct.add_argument("--device", default="cuda")
    reconstruct.add_argument("--split", choices=["validation", "test"], default="validation")
    reconstruct.add_argument("--text-tokens", type=int, default=512)
    reconstruct.add_argument("--max-samples", type=int, default=20)
    reconstruct.add_argument("--with-loss", action="store_true")
    reconstruct.set_defaults(handler=_reconstruct)
    sample = commands.add_parser("sample")
    for name in ("checkpoint", "codec", "config", "output"):
        sample.add_argument(f"--{name}", required=True)
    sample.add_argument("--tokenizer")
    sample.add_argument("--device", default="cuda")
    sample.add_argument("--text-tokens", type=int, default=512)
    sample.add_argument("--num-samples", type=int, default=20)
    sample.add_argument("--batch-size", type=int, default=1)
    sample.add_argument("--seed", type=int, default=3)
    sample.add_argument("--no-ema", action="store_true")
    sample.set_defaults(handler=_sample)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--model", help="Local causal LM for PPL; omit for repetition metrics only")
    evaluate.add_argument("--context-window", type=int, default=1024)
    evaluate.add_argument("--stride", type=int, default=512)
    evaluate.add_argument("--device", default="cuda")
    evaluate.set_defaults(handler=_evaluate)
    judge = commands.add_parser("judge-prepare")
    for name in ("a", "b", "output"):
        judge.add_argument(f"--{name}", required=True)
    judge.add_argument("--text-tokens", type=int)
    judge.add_argument("--judge-tokenizer")
    judge.add_argument("--context-budget", type=int)
    judge.add_argument("--response-reserve", type=int, default=2048)
    judge.set_defaults(handler=_judge_prepare)
    collect = commands.add_parser("judge-collect")
    for name in ("packets", "responses", "output"):
        collect.add_argument(f"--{name}", required=True)
    def collect_results(args):
        from .judge import collect_judge_results
        return collect_judge_results(args.packets, args.responses, args.output)
    collect.set_defaults(handler=collect_results)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
