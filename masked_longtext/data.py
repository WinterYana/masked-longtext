"""Local, document-preserving preparation. This module never downloads a corpus."""
from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield record


class DatasetWriter:
    """Bounded row buffers, .npy shards and auditable row-to-document mappings."""

    def __init__(self, output, *, kind, vocab_size, tokenizer_info, shard_rows=256, extra=None):
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Output directory must be empty: {self.output}")
        if kind not in ("text_tokens", "vq_codes") or vocab_size < 2 or shard_rows < 1:
            raise ValueError("Invalid dataset kind, vocab_size or shard_rows")
        self.output.mkdir(parents=True, exist_ok=True)
        self.shard_rows = int(shard_rows)
        self.manifest = {"format_version": 1, "kind": kind, "vocab_size": int(vocab_size),
                         "tokenizer": tokenizer_info, "splits": {}, "stats": {}, **(extra or {})}
        self.buffers = {}

    def add(self, split, values, *, doc_id, start):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split {split}")
        ids = np.asarray(values)
        if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu":
            raise ValueError("Expected nonempty one-dimensional integer IDs")
        if int(ids.min()) < 0 or int(ids.max()) >= self.manifest["vocab_size"]:
            raise ValueError("Token/code ID out of vocabulary range")
        if self.manifest["vocab_size"] > np.iinfo(np.int32).max:
            raise ValueError("Vocabulary exceeds supported int32 storage")
        key = (split, len(ids))
        entries = self.buffers.setdefault(key, [])
        entries.append((ids.astype(np.int32), {"doc_id": str(doc_id), "start": int(start)}))
        if len(entries) >= self.shard_rows:
            self._flush(key)

    def _flush(self, key):
        entries = self.buffers.get(key, [])
        if not entries:
            return
        split, length = key
        group = self.manifest["splits"].setdefault(split, {}).setdefault(str(length), {"rows": 0, "shards": []})
        stem = f"{split}_{length}_{len(group['shards']):05d}"
        array_path = self.output / f"{stem}.npy"
        meta_path = self.output / f"{stem}.jsonl"
        np.save(array_path, np.stack([entry[0] for entry in entries]), allow_pickle=False)
        with meta_path.open("w", encoding="utf-8") as handle:
            for _, meta in entries:
                handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        group["shards"].append({"path": array_path.name, "metadata": meta_path.name, "rows": len(entries),
                                 "sha256": file_hash(array_path), "metadata_sha256": file_hash(meta_path)})
        group["rows"] += len(entries)
        self.buffers[key] = []

    def finish(self, stats=None):
        for key in list(self.buffers):
            self._flush(key)
        self.manifest["stats"] = stats or {}
        self.manifest["dataset_id"] = canonical_hash(self.manifest)
        write_json(self.output / "manifest.json", self.manifest)
        return self.manifest


def read_manifest(path):
    path = Path(path)
    path = path / "manifest.json" if path.is_dir() else path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    claimed = manifest.get("dataset_id")
    content = {k: v for k, v in manifest.items() if k != "dataset_id"}
    if manifest.get("format_version") != 1 or claimed != canonical_hash(content):
        raise ValueError("Invalid dataset manifest identity or unsupported format")
    if "documents_sha256" in manifest and file_hash(path.parent / "documents.jsonl") != manifest["documents_sha256"]:
        raise ValueError("Dataset document audit checksum mismatch")
    return manifest


def _contained_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Shard path leaves the dataset directory")
    return path


class SequenceStore:
    """Memory-mapped fixed-length rows; no pickle loading and no data downloads."""

    def __init__(self, path, split, length, *, expected_kind=None, verify_hashes=True):
        path = Path(path)
        self.root = path if path.is_dir() else path.parent
        self.manifest = read_manifest(path)
        if expected_kind and self.manifest["kind"] != expected_kind:
            raise ValueError(f"Expected {expected_kind}, found {self.manifest['kind']}")
        self.split, self.length = split, int(length)
        self.arrays, self.shards, self.ends = [], [], []
        group = self.manifest["splits"].get(split, {}).get(str(length), {"rows": 0, "shards": []})
        total = 0
        for shard in group["shards"]:
            array_path = _contained_file(self.root, shard["path"])
            meta_path = _contained_file(self.root, shard["metadata"])
            if verify_hashes and (file_hash(array_path) != shard["sha256"] or file_hash(meta_path) != shard["metadata_sha256"]):
                raise ValueError(f"Dataset shard checksum mismatch: {shard['path']}")
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if array.shape != (shard["rows"], self.length) or array.dtype.kind not in "iu":
                raise ValueError(f"Invalid shard shape/type: {array_path}")
            self.arrays.append(array)
            self.shards.append(shard)
            total += len(array)
            self.ends.append(total)
        if total != group["rows"]:
            raise ValueError("Manifest row count mismatch")

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        start = self.ends[shard_index - 1] if shard_index else 0
        return np.array(self.arrays[shard_index][index - start], dtype=np.int64, copy=True)

    def iter_records(self):
        for array, shard in zip(self.arrays, self.shards):
            metadata = list(read_jsonl(_contained_file(self.root, shard["metadata"])))
            if len(metadata) != len(array):
                raise ValueError("Row metadata count mismatch")
            for row, meta in zip(array, metadata):
                yield np.array(row, dtype=np.int64, copy=True), meta


def prepare_jsonl(input_path, output, *, lengths, vocab_size, tokenizer_info, tokenizer=None,
                  seed=42, val_fraction=.01, test_fraction=.01, shard_rows=256, eos_id=None):
    """One input record is one document; splits precede contiguous nonoverlap windows.

    Records contain id/doc_id and input_ids, or text with an explicitly supplied
    tokenizer callable. No code/content filtering is silently applied.
    """
    lengths = sorted(set(map(int, lengths)))
    if not lengths or any(n < 4 or n % 4 for n in lengths):
        raise ValueError("Text lengths must be positive multiples of four")
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Invalid split fractions")
    if eos_id is not None and not 0 <= eos_id < vocab_size:
        raise ValueError("EOS out of vocabulary range")
    writer = DatasetWriter(output, kind="text_tokens", vocab_size=vocab_size, tokenizer_info=tokenizer_info,
                           shard_rows=shard_rows, extra={"lengths": lengths, "split_seed": seed,
                           "split_fractions": {"validation": val_fraction, "test": test_fraction},
                           "source_sha256": file_hash(input_path), "eos_id": eos_id})
    stats = {"documents": 0, "duplicate_documents": 0, "empty_documents": 0,
             "tokens": 0, "discarded_tail_tokens_by_length": {str(n): 0 for n in lengths}}
    seen_content, seen_ids = set(), set()
    with (Path(output) / "documents.jsonl").open("w", encoding="utf-8") as audit:
        for line_index, record in enumerate(read_jsonl(input_path)):
            doc_id = str(record.get("doc_id", record.get("id", line_index)))
            if doc_id in seen_ids:
                raise ValueError(f"Repeated document ID: {doc_id}")
            seen_ids.add(doc_id)
            if "input_ids" in record:
                ids = record["input_ids"]
            elif isinstance(record.get("text"), str) and tokenizer is not None:
                ids = list(tokenizer(record["text"]))
            else:
                raise ValueError("Each document needs input_ids or text and a local tokenizer")
            if not isinstance(ids, list) or any(type(v) is not int or not 0 <= v < vocab_size for v in ids):
                raise ValueError(f"Invalid integer ID or vocabulary range in {doc_id}")
            if not ids:
                stats["empty_documents"] += 1
                continue
            if eos_id is not None and ids[-1] != eos_id:
                ids = ids + [eos_id]
            identity = canonical_hash(ids)
            if identity in seen_content:
                stats["duplicate_documents"] += 1
                continue
            seen_content.add(identity)
            u = int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()[:16], 16) / 2**64
            split = "test" if u < test_fraction else "validation" if u < test_fraction + val_fraction else "train"
            audit.write(json.dumps({"doc_id": doc_id, "content_hash": identity, "split": split,
                                    "tokens": len(ids)}, ensure_ascii=False) + "\n")
            stats["documents"] += 1
            stats["tokens"] += len(ids)
            for length in lengths:
                stats["discarded_tail_tokens_by_length"][str(length)] += len(ids) % length
                for start in range(0, len(ids) - length + 1, length):
                    writer.add(split, ids[start:start + length], doc_id=doc_id, start=start)
    writer.manifest["documents_sha256"] = file_hash(Path(output) / "documents.jsonl")
    return writer.finish(stats)
