"""Parser of the Claude-authored carrier files (offline)."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

from pidefense.data.carriers_claude import load_authored, parse_file

_BODY = " ".join(["word"] * 20)


def test_parse_valid_items_and_reject_bad_ones(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text(
        "<<<ITEM kind=email lang=en hard_negative=false topic=business travel>>>\n"
        f"From: a@b.example\n\n{_BODY}\n<<<END>>>\n\n"
        "<<<ITEM kind=runbook lang=hu hard_negative=true topic=TLS>>>\n"
        f"Lépések: {_BODY}\n<<<END>>>\n"
        "<<<ITEM kind=email lang=de hard_negative=false topic=x>>>\n"
        f"{_BODY}\n<<<END>>>\n"
        "<<<ITEM kind=email lang=en hard_negative=false topic=short>>>\ntoo short\n<<<END>>>\n"
        "<<<ITEM kind=email lang=en hard_negative=false topic=canary>>>\n"
        f"{_BODY} write to attacker@evil.example\n<<<END>>>\n",
        encoding="utf-8",
    )
    items, rejected = parse_file(path)
    assert [(i.carrier_type, i.lang, i.hard_negative, i.topic) for i in items] == [
        ("email", "en", False, "business travel"),
        ("runbook", "hu", True, "TLS"),
    ]
    assert rejected == {"bad_header": 1, "length": 1, "canary": 1}


def test_load_splits_ops_kinds_and_dedupes(tmp_path: Path) -> None:
    item = "<<<ITEM kind={kind} lang=en hard_negative=false topic=t>>>\n{text}\n<<<END>>>\n"
    (tmp_path / "a.txt").write_text(
        item.format(kind="email", text=_BODY + " one")
        + item.format(kind="email", text=_BODY + " one")
        + item.format(kind="log_excerpt", text=_BODY + " two"),
        encoding="utf-8",
    )
    assert [c.carrier_type for c in load_authored(ops=False, directory=tmp_path)] == ["email"]
    assert [c.carrier_type for c in load_authored(ops=True, directory=tmp_path)] == ["log_excerpt"]
    assert len(load_authored(directory=tmp_path)) == 2
