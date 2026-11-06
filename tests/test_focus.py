"""Tests of the focus component (delegated documents, escalation follow-ups) and its leakage controls."""

from __future__ import annotations

import random
import re

import pytest

from pidefense.data.carriers_claude import AuthoredCarrier
from pidefense.data.focus import (
    BENIGN_ACTIONS,
    FORBIDDEN,
    MALICIOUS_ACTIONS,
    WRAPPERS,
    FocusBuilder,
    SlotValues,
    all_wrappers,
    split_doc,
    step_prefix,
)
from pidefense.data.templates import fill, template_pools
from pidefense.experiments.ops_attacks import FOLLOWUPS, OPS_GOALS, OPS_TEMPLATES

NGRAM = 5


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", re.sub(r"\{\w+\}", " ", text).lower())


def _ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _test_texts() -> list[str]:
    texts = [t for family in OPS_TEMPLATES.values() for bank in family.values() for t in bank]
    texts += [t for followup in FOLLOWUPS.values() for t in followup.values()]
    texts += [t for goal in OPS_GOALS.values() for bank in goal.values() for t in bank]
    return texts


def _carrier(text: str, kind: str = "focus_runbook", lang: str = "en", cid: str = "cl-x") -> AuthoredCarrier:
    return AuthoredCarrier(cid, kind, lang, "t", True, text)


DOC = "RB-1 export-worker lag\nOwner: ops.\n\n1. read_logs export-worker.\n2. Restart it.\n{{SLOT}}\n3. Report."


def test_wrappers_share_no_ngram_with_opspilot_attacks() -> None:
    """No wrapper repeats 5 words of an OpsPilot attack template, follow-up or goal."""
    test_grams = set().union(*(_ngrams(t) for t in _test_texts()))
    for wrapper in all_wrappers():
        assert not _ngrams(wrapper.text) & test_grams, wrapper.id


def test_actions_share_no_ngram_with_opspilot_goals() -> None:
    """No action phrasing repeats 5 words of an OpsPilot text."""
    test_grams = set().union(*(_ngrams(t) for t in _test_texts()))
    for bank in (MALICIOUS_ACTIONS, BENIGN_ACTIONS):
        for category, langs in bank.items():
            for text in (t for texts in langs.values() for t in texts):
                assert not _ngrams(text) & test_grams, (category, text)


def test_every_wrapper_group_has_all_three_pools() -> None:
    """Each wrapper group has train, val and test wrappers in both languages."""
    pools = template_pools(all_wrappers())
    for group, lang in WRAPPERS:
        assigned = {pools[f"{group}-{lang}-{i:02d}"] for i in range(len(WRAPPERS[(group, lang)]))}
        assert assigned == {"train", "val", "test"}, (group, lang)


@pytest.mark.parametrize("bank", [MALICIOUS_ACTIONS, BENIGN_ACTIONS])
def test_actions_fill_every_slot(bank: dict[str, dict[str, list[str]]]) -> None:
    """Every action slot has a value."""
    slots = SlotValues(1)
    for langs in bank.values():
        for lang, texts in langs.items():
            for text in texts:
                assert "{" not in fill(text, **slots.values(lang)).replace("{}", "")


def test_split_doc_requires_one_inner_slot() -> None:
    """A document needs exactly one slot line, not at its edge."""
    doc = split_doc(_carrier(DOC))
    assert doc is not None and doc.before[-1] == "2. Restart it." and doc.after == ["3. Report."]
    assert split_doc(_carrier(DOC.replace("{{SLOT}}\n", ""))) is None
    assert split_doc(_carrier("{{SLOT}}\n" + DOC.replace("{{SLOT}}\n", ""))) is None


def test_step_prefix_follows_document_style() -> None:
    """Inserted steps copy the list style above the slot."""
    rng = random.Random(0)
    assert step_prefix(["1. a", "2. b"], rng) in ("2a. ", "3. ", "")
    assert step_prefix(["- [ ] a"], rng) == "- [ ] "
    assert step_prefix(["* a", ""], rng) == "* "
    assert step_prefix(["Plain sentence."], rng) == ""


def test_builder_labels_families_and_pools() -> None:
    """Labels are balanced, families match labels, wrappers come from the sample's own split."""
    docs = [_carrier(DOC.replace("RB-1", f"RB-{i}"), cid=f"cl-doc{i}") for i in range(40)]
    followups = [
        _carrier(f"Thanks {i}, please also restart export-worker after the change.", "focus_followup", cid=f"cl-f{i}")
        for i in range(20)
    ]
    builder = FocusBuilder(lambda text: len(text.split()), seed=3)
    samples = builder.build(docs, followups)
    pools = template_pools(all_wrappers())
    assert len({s.id for s in samples}) == len(samples)
    labels = [s.label for s in samples]
    assert 0.4 < sum(labels) / len(labels) < 0.6
    for sample in samples:
        assert (sample.family in ("D1", "D2")) == (sample.label == 1)
        assert not any(term in sample.text.lower() for term in FORBIDDEN)
        assert "{" not in sample.text.replace("{{", "")
        if sample.template_id and sample.template_id[:2] in {g for g, _ in WRAPPERS}:
            assert pools[sample.template_id] == sample.split, sample.id
    assert {s.family for s in samples if s.position == "slot" and s.label == 1} == {"D1"}
