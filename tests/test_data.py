import json

import pytest

from masked_longtext.data import SequenceStore, prepare_jsonl, read_manifest


def test_windows_are_contiguous_and_document_splits_do_not_leak(tmp_path):
    source = tmp_path / "docs.jsonl"
    records = [{"id": f"d{i}", "input_ids": list(range(i * 20, i * 20 + 17))} for i in range(50)]
    source.write_text("\n".join(json.dumps(x) for x in records), encoding="utf-8")
    target = tmp_path / "data"
    manifest = prepare_jsonl(source, target, lengths=[4, 8], vocab_size=2000,
                            tokenizer_info={"name": "test", "revision": "1"}, val_fraction=.2, test_fraction=.2)
    assert manifest["kind"] == "text_tokens"
    locations = {}
    for split in ("train", "validation", "test"):
        for length in (4, 8):
            store = SequenceStore(target, split, length)
            for ids, meta in store.iter_records():
                original = records[int(meta["doc_id"][1:])]["input_ids"]
                assert ids.tolist() == original[meta["start"]:meta["start"] + length]
                locations.setdefault(meta["doc_id"], set()).add(split)
    assert all(len(splits) == 1 for splits in locations.values())


def test_duplicate_documents_are_removed_and_manifest_tampering_fails(tmp_path):
    source = tmp_path / "docs.jsonl"
    source.write_text('\n'.join(json.dumps({"id": n, "input_ids": list(range(12))}) for n in ("a", "b")), encoding="utf-8")
    target = tmp_path / "data"
    manifest = prepare_jsonl(source, target, lengths=[4], vocab_size=20,
                            tokenizer_info={"name": "test"}, val_fraction=0, test_fraction=0)
    assert manifest["stats"]["duplicate_documents"] == 1
    assert len(SequenceStore(target, "train", 4)) == 3
    path = target / "manifest.json"
    obj = json.loads(path.read_text())
    obj["vocab_size"] = 21
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="identity"):
        read_manifest(target)


def test_invalid_ids_and_nonempty_output_are_rejected(tmp_path):
    source = tmp_path / "docs.jsonl"
    source.write_text('{"id":"a","input_ids":[1,2,999,4]}', encoding="utf-8")
    with pytest.raises(ValueError, match="range"):
        prepare_jsonl(source, tmp_path / "bad", lengths=[4], vocab_size=20, tokenizer_info={"name": "test"})
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        prepare_jsonl(source, existing, lengths=[4], vocab_size=20, tokenizer_info={"name": "test"})


def test_document_audit_tampering_is_rejected(tmp_path):
    source = tmp_path / "docs.jsonl"
    source.write_text('{"id":"a","input_ids":[1,2,3,4,5,6,7,8]}', encoding="utf-8")
    target = tmp_path / "data"
    prepare_jsonl(source, target, lengths=[4], vocab_size=20, tokenizer_info={"name": "test"},
                  val_fraction=0, test_fraction=0)
    (target / "documents.jsonl").write_text('{"doc_id":"a","split":"validation"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="audit checksum"):
        SequenceStore(target, "train", 4)


def test_content_split_is_stable_when_document_ids_and_order_change(tmp_path):
    first = [{"id": f"a{i}", "input_ids": [i, i + 1, i + 2, i + 3]} for i in range(30)]
    second = [{"id": f"renamed{i}", "input_ids": row["input_ids"]} for i, row in enumerate(reversed(first))]
    memberships = []
    for index, records in enumerate((first, second)):
        source = tmp_path / f"source{index}.jsonl"
        source.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        target = tmp_path / f"data{index}"
        prepare_jsonl(source, target, lengths=[4], vocab_size=100, tokenizer_info={"name": "test"},
                      seed=4, val_fraction=.2, test_fraction=.2)
        memberships.append({tuple(ids.tolist()): split for split in ("train", "validation", "test")
                            for ids, _ in SequenceStore(target, split, 4).iter_records()})
    assert memberships[0] == memberships[1]
