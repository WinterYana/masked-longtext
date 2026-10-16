import json
from types import SimpleNamespace

import pytest

from masked_longtext import __main__ as cli


def _args(tmp_path, lengths=(512,), declared=512):
    records = [{"id": i, "text": "A short test.", "text_token_count": length} for i, length in enumerate(lengths)]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
    b.write_text(a.read_text(encoding="utf-8"), encoding="utf-8")
    return SimpleNamespace(a=str(a), b=str(b), output=tmp_path / "judge", text_tokens=declared,
                           judge_tokenizer="local-judge", context_budget=500, response_reserve=100)


def test_judge_budget_uses_rendered_chat_not_plain_text(tmp_path, monkeypatch):
    calls = []
    class Tokenizer:
        chat_template = "test"
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert isinstance(messages, list) and messages[0]["role"] == "system"
            assert tokenize and add_generation_prompt
            calls.append(messages)
            return list(range(200))
        def encode(self, *args, **kwargs):
            raise AssertionError("Chat budget must not use plain encode")
    monkeypatch.setattr(cli, "local_tokenizer", lambda _: (Tokenizer(), {}))
    result = cli._judge_prepare(_args(tmp_path))
    assert len(calls) == 2 and result["token_budget_status"] == "checked"


@pytest.mark.parametrize("lengths,declared", [((512,), 8192), ((512, 8192), None)])
def test_judge_rejects_misdeclared_and_mixed_lengths(tmp_path, lengths, declared):
    with pytest.raises(ValueError, match="length"):
        cli._judge_prepare(_args(tmp_path, lengths, declared))


def test_all_main_stages_are_available_without_execution():
    parser = cli.parser()
    for command in ("prepare", "train-codec", "encode", "reconstruct", "train-prior", "sample", "evaluate", "judge-prepare", "judge-collect"):
        assert command in parser.format_help()
