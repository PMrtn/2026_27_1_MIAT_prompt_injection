"""Dataset generator: templates, obfuscation, embedding and leakage controls (offline)."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import base64
import collections
import random

from pidefense.attacks.goals import ATTACKER_EMAIL, ATTACKER_IBAN, MISINFO_CODE, PHISHING_URL
from pidefense.data.build import Carrier, DatasetBuilder, _contains_canary
from pidefense.data.carriers_faker import generate_faker_carriers
from pidefense.data.embed import embed, wrap
from pidefense.data.obfuscation import _ZERO_WIDTH, obfuscate
from pidefense.data.templates import E2E_GOALS, TRAIN_GOALS, all_templates, template_pools
from pidefense.schemas import SEEN_FAMILIES, UNSEEN_FAMILIES


def test_templates_have_slots_and_unique_ids() -> None:
    templates = all_templates()
    assert len({t.id for t in templates}) == len(templates)
    for template in templates:
        slot = "{payload}" if template.family == "U2" else "{goal}"
        assert slot in template.text, template.id


def test_template_pools_are_stratified() -> None:
    templates = all_templates()
    pools = template_pools(templates)
    for family in SEEN_FAMILIES:
        for lang in ("en", "hu"):
            members = [pools[t.id] for t in templates if t.family == family and t.lang == lang]
            assert {"train", "val", "test"} <= set(members), (family, lang)
            assert members.count("train") >= len(members) * 0.6
    assert all(pools[t.id] == "test" for t in templates if t.family in UNSEEN_FAMILIES)


def test_canaries_only_in_e2e_goals() -> None:
    train_text = " ".join(goal for goals in TRAIN_GOALS.values() for goal in goals)
    assert not _contains_canary(train_text)
    e2e_text = " ".join(g for goal in E2E_GOALS.values() for phrasings in goal.values() for g in phrasings)
    for canary in (ATTACKER_EMAIL, PHISHING_URL, MISINFO_CODE):
        assert canary in e2e_text
    assert _contains_canary(f"pay to {ATTACKER_IBAN.replace(' ', '')}")


def test_obfuscation_transforms() -> None:
    rng = random.Random(0)
    text = "send the inbox to someone"
    assert base64.b64decode(obfuscate("base64", text, rng)).decode() == text
    assert obfuscate("reversed", obfuscate("reversed", text, rng), rng) == text
    hidden = obfuscate("zero_width", text, rng)
    assert hidden != text and "".join(ch for ch in hidden if ch not in _ZERO_WIDTH) == text
    assert "a =" in obfuscate("split", text, rng, "en") and "Legyen" in obfuscate("split", text, rng, "hu")


def test_embed_positions_and_wrappers() -> None:
    rng = random.Random(1)
    carrier = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    assert embed(carrier, "PAYLOAD", "start", "plain", rng).startswith("PAYLOAD")
    assert embed(carrier, "PAYLOAD", "end", "plain", rng).endswith("PAYLOAD")
    middle = embed(carrier, "PAYLOAD", "middle", "plain", rng)
    assert not middle.startswith("PAYLOAD") and not middle.endswith("PAYLOAD") and "PAYLOAD" in middle
    assert wrap("x", "html_comment") == "<!-- x -->"
    assert wrap("x", "hidden_span").startswith('<span style="display:none">')


def test_faker_carrier_slot_is_json_safe() -> None:
    carrier = next(c for c in generate_faker_carriers({"hu": 5}) if c.is_json)
    injected = carrier.fill_slot('say "hi"\nnow')
    assert '\\"hi\\"' in injected and "\\n" in injected
    assert carrier.benign_value in carrier.fill_slot(None)


def _small_build() -> list:
    carriers = [
        Carrier(f"llm-{i:04d}", "email", "en" if i % 3 else "hu", i % 5 == 0, "test", text=f"Hello team. Item {i}.")
        for i in range(400)
    ]
    builder = DatasetBuilder(length_fn=lambda text: len(text.split()), seed=3)
    return builder.build(carriers, standalone={"train": 10, "val": 5, "test": 16})


def test_build_respects_splits_and_pools() -> None:
    samples = _small_build()
    pools = template_pools()
    by_split = collections.Counter(s.split for s in samples)
    assert set(by_split) == {"train", "val", "test"}
    for sample in samples:
        assert not _contains_canary(sample.text)
        if sample.label == 0:
            assert sample.family == "benign" and sample.template_id is None
            continue
        assert sample.template_id is not None
        if sample.family in UNSEEN_FAMILIES:
            assert sample.split == "test"
        else:
            assert pools[sample.template_id] == sample.split
    carrier_splits: dict[str, set[str]] = collections.defaultdict(set)
    for sample in samples:
        if sample.carrier_id:
            carrier_splits[sample.carrier_id].add(sample.split)
    assert all(len(splits) == 1 for splits in carrier_splits.values())
