import importlib
import json

import pytest


CRITERIA = ["global_coherence", "entity_consistency", "progression", "non_redundancy", "language_quality", "prompt_adherence"]


def api():
    return importlib.import_module("masked_longtext.judge")


def records():
    return [{"pair_id": "example", "A": "Alice arrived.\n\nAlice left.", "B": "Bob arrived.\n\nBob stayed.",
             "metadata": {"system": "SECRET-MODEL", "ppl": 12}}]


def prepare(tmp_path, **kwargs):
    result = api().prepare_judge_packets(records(), tmp_path, length_condition=512, **kwargs)
    packets = [json.loads(line) for line in (tmp_path / "packets.jsonl").read_text().splitlines()]
    return result, packets


def response(winner="A", quote="Alice arrived.", segment="A:p001"):
    side = {"scores": {k: (None if k == "prompt_adherence" else 4) for k in CRITERIA},
            "evidence": [], "unassessable": [{"criterion": "prompt_adherence", "reason": "No source prompt."}]}
    value = {"coverage": "complete", "A": json.loads(json.dumps(side)), "B": json.loads(json.dumps(side)),
             "winner": winner, "justification": "More consistent.", "limitations": []}
    if quote:
        value["A"]["evidence"] = [{"criterion": "global_coherence", "segments": [segment], "quotes": [quote], "issue": "Consistent arrival."}]
    return value


def collect(tmp_path, packets, responses):
    path = tmp_path / "responses.jsonl"
    path.write_text("\n".join(json.dumps({"packet_id": p["packet_id"], "response": r})
                              for p, r in zip(packets, responses)), encoding="utf-8")
    return api().collect_judge_results(tmp_path, path)


def test_blinding_preserves_full_text_and_swaps_stable_ids(tmp_path):
    _, packets = prepare(tmp_path)
    assert len(packets) == 2
    assert packets[0]["packet_id"] != packets[1]["packet_id"]
    for packet in packets:
        prompt = json.dumps(packet["messages"])
        assert "SECRET-MODEL" not in prompt and '"ppl"' not in prompt and "example" not in prompt
        assert "Alice arrived." in prompt and "Alice left." in prompt and "Bob stayed." in prompt
    assert "A:p001" in packets[0]["messages"][1]["content"]
    assert "A:p001" in packets[1]["messages"][1]["content"]
    mapping = json.loads((tmp_path / "mapping.json").read_text())
    assert mapping["packets"][packets[0]["packet_id"]]["display_to_original"] == {"A": "A", "B": "B"}
    assert mapping["packets"][packets[1]["packet_id"]]["display_to_original"] == {"A": "B", "B": "A"}
    _, repeated = prepare(tmp_path / "repeat")
    assert packets == repeated


def test_judge_export_preserves_existing_output(tmp_path):
    sentinel = tmp_path / "mapping.json"
    sentinel.write_text("existing experiment", encoding="utf-8")
    with pytest.raises(FileExistsError, match="empty"):
        api().prepare_judge_packets(records(), tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "existing experiment"
    assert not (tmp_path / "packets.jsonl").exists()


def test_both_orders_map_to_original_winner_and_keep_missing(tmp_path):
    _, packets = prepare(tmp_path)
    result = collect(tmp_path, packets, [response("A"), response("B", "Bob arrived.")])
    assert result["counts"]["A_wins"] == 1
    assert result["pairs"][0]["outcome"] == "A"
    missing = collect(tmp_path, packets[:1], [response("A")])
    assert missing["counts"]["missing_responses"] == 1
    assert missing["counts"]["missing_pairs"] == 1
    assert missing["counts"]["planned_pairs"] == 1


@pytest.mark.parametrize("mutation", ["float", "boolean", "bad_quote", "wrong_segment", "extra", "missing_reason", "null_conditional"])
def test_strict_response_validation_rejects_invalid_evidence_and_scores(tmp_path, mutation):
    if mutation == "null_conditional":
        data = records()
        data[0]["prompt"] = "Continue the arrival."
        api().prepare_judge_packets(data, tmp_path)
        packets = [json.loads(x) for x in (tmp_path / "packets.jsonl").read_text().splitlines()]
    else:
        _, packets = prepare(tmp_path)
    value = response()
    if mutation == "float": value["A"]["scores"]["global_coherence"] = 4.0
    if mutation == "boolean": value["A"]["scores"]["global_coherence"] = True
    if mutation == "bad_quote": value["A"]["evidence"][0]["quotes"] = ["Alice flew away."]
    if mutation == "wrong_segment": value["A"]["evidence"][0]["segments"] = ["A:p002"]
    if mutation == "extra": value["hidden"] = "extra"
    if mutation == "missing_reason": value["A"]["unassessable"] = []
    if mutation == "null_conditional": value["A"]["unassessable"] = []
    result = collect(tmp_path, packets, [value, response("B", "Bob arrived.")])
    assert result["counts"]["invalid_responses"] >= 1
    assert result["pairs"][0]["outcome"] == "invalid"
    assert result["counts"]["A_wins"] == 0


def test_order_disagreement_incomplete_and_raw_malformed_json_are_retained(tmp_path):
    _, packets = prepare(tmp_path)
    result = collect(tmp_path, packets, [response("A"), response("A", "Bob arrived.")])
    assert result["counts"]["order_inconsistent_pairs"] == 1
    value = response()
    value["coverage"] = "incomplete"
    result = collect(tmp_path, packets, [value, response("B", "Bob arrived.")])
    assert result["counts"]["incomplete_pairs"] == 1
    result = collect(tmp_path, packets, ["{ malformed", response("B", "Bob arrived.")])
    assert result["counts"]["invalid_responses"] == 1
    assert result["responses"][0]["errors"]


def test_budget_rejects_without_truncating_or_writing_packets(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        prepare(tmp_path, token_counter=lambda messages: 5000, context_budget=4000)
    assert not (tmp_path / "packets.jsonl").exists()
    with pytest.raises(ValueError, match="token_counter"):
        prepare(tmp_path, context_budget=4000)


def test_duplicate_pair_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        api().prepare_judge_packets(records() * 2, tmp_path)


def test_exact_segments_reconstruct_candidates_without_editing(tmp_path):
    text = "  first\r\n\r\nSecond\n\nthird\n"
    api().prepare_judge_packets([{"pair_id": "x", "A": text, "B": "a" * 2500}], tmp_path, segment_chars=1000)
    mapping = json.loads((tmp_path / "mapping.json").read_text())
    first = next(iter(mapping["packets"].values()))
    for display, original in [("A", text), ("B", "a" * 2500)]:
        assert first["texts"][display] == original
        assert "".join(s["text"] for s in first["segments"][display]) == original
        for s in first["segments"][display]:
            assert original[s["start"]:s["end"]] == s["text"]


def test_ties_and_uncertain_use_all_planned_pairs_denominator(tmp_path):
    _, packets = prepare(tmp_path)
    result = collect(tmp_path, packets, [response("tie"), response("tie", "Bob arrived.")])
    assert result["counts"]["ties"] == 1
    assert result["fractions_of_all_planned_pairs"]["tie"] == 1.0
    result = collect(tmp_path, packets, [response("uncertain"), response("B", "Bob arrived.")])
    assert result["counts"]["uncertain_pairs"] == 1
    assert result["counts"]["A_wins"] == 0


def test_duplicate_responses_invalidate_packet_instead_of_picking_a_winner(tmp_path):
    _, packets = prepare(tmp_path)
    path = tmp_path / "responses.jsonl"
    rows = [{"packet_id": packets[0]["packet_id"], "response": response("A")},
            {"packet_id": packets[0]["packet_id"], "response": response("B")},
            {"packet_id": packets[1]["packet_id"], "response": response("B", "Bob arrived.")}]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = api().collect_judge_results(tmp_path, path)
    assert result["counts"]["duplicate_responses"] == 1
    assert result["counts"]["invalid_responses"] == 2
    assert result["pairs"][0]["outcome"] == "invalid"


def test_malformed_envelopes_and_unknown_ids_are_retained_without_becoming_wins(tmp_path):
    prepare(tmp_path)
    path = tmp_path / "responses.jsonl"
    path.write_text('{bad JSON\n' + json.dumps({"packet_id": "unknown", "response": response()}) + '\n', encoding="utf-8")
    result = api().collect_judge_results(tmp_path, path)
    assert result["counts"]["invalid_responses"] == 2
    assert result["counts"]["unexpected_responses"] == 1
    assert result["counts"]["missing_responses"] == 2
    assert result["counts"]["missing_pairs"] == 1
    assert result["provenance"]["responses_path"] == str(path.resolve())


def test_duplicate_json_keys_are_invalid_not_last_value_wins(tmp_path):
    _, packets = prepare(tmp_path)
    raw = json.dumps(response())
    raw = raw.replace('"winner": "A"', '"winner": "B", "winner": "A"')
    result = collect(tmp_path, packets, [raw, response("B", "Bob arrived.")])
    assert result["counts"]["invalid_responses"] == 1


def test_hash_tampering_rejected_before_response_collection(tmp_path):
    _, packets = prepare(tmp_path)
    packets[0]["messages"][1]["content"] += "changed"
    (tmp_path / "packets.jsonl").write_text("\n".join(json.dumps(p) for p in packets), encoding="utf-8")
    path = tmp_path / "responses.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        api().collect_judge_results(tmp_path, path)


def test_collection_output_cannot_overwrite_raw_answers(tmp_path):
    prepare(tmp_path)
    path = tmp_path / "responses.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        api().collect_judge_results(tmp_path, path, output_path=path)
    assert path.read_text() == ""
