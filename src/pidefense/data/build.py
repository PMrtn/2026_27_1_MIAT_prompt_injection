"""Assemble the detector dataset from carriers and attack templates.

Leakage controls:

* carriers are split 70/15/15 into train/val/test pools by a hash of their id; every sample built from a carrier
  inherits the carrier's split;
* seen-family templates are split 70/15/15 per (family, lang) (``template_pools``); an injected sample only uses
  templates from its own split's pool; unseen families (U1-U3) appear only in the test split;
* attacker slot values are random Faker values; the end-to-end canaries never appear (checked, the build fails);
* every sample fits in ``MAX_TOKENS`` model tokens, so training-time truncation never cuts off a payload;
* exact duplicates (after normalisation) are dropped.

Usage:
    python -m pidefense.data.build
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from faker import Faker

from pidefense.attacks.goals import ATTACKER_DOMAIN, ATTACKER_EMAIL, ATTACKER_IBAN, MISINFO_CODE, normalize_iban
from pidefense.config import REPO_ROOT
from pidefense.data import carriers_llm
from pidefense.data.carriers_faker import FakerCarrier, generate_faker_carriers
from pidefense.data.embed import choose_position_and_wrapper, embed
from pidefense.data.obfuscation import obfuscate
from pidefense.data.templates import TRAIN_GOALS, U2_BASE, Template, all_templates, fill, template_pools
from pidefense.logging_utils import setup_logging
from pidefense.schemas import UNSEEN_FAMILIES, Sample

logger = logging.getLogger(__name__)

DATASET_DIR = REPO_ROOT / "data" / "dataset"
MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"
MAX_TOKENS = 256
SPLITS = ("train", "val", "test")
SEEN = ("S1", "S2", "S3", "S4", "S5")
_FILE_DIRS = ("docs", "finance", "hr", "projects", "backup", "shared", "contracts", "reports")
_FILE_EXTS = ("pdf", "docx", "xlsx", "txt", "csv", "pptx")
_MODEL_SUFFIXES = ("X200", "Pro", "Air", "Max", "Lite", "S", "Plus", "Mini")


@dataclass
class Carrier:
    """A benign host text: free text (LLM) or structured with one slot (Faker)."""

    carrier_id: str
    carrier_type: str
    lang: str
    hard_negative: bool
    generator: str
    text: str = ""
    faker: FakerCarrier | None = None

    @property
    def benign_text(self) -> str:
        """The carrier without any payload."""
        return self.faker.fill_slot(None) if self.faker else self.text


def carrier_pool(carrier_id: str) -> str:
    """Deterministic 70/15/15 split of a carrier."""
    bucket = int(hashlib.sha256(f"carrier:{carrier_id}".encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else ("val" if bucket < 85 else "test")


def load_carriers(faker_counts: dict[str, int]) -> list[Carrier]:
    """LLM carriers from the generated JSONL plus freshly built Faker carriers (deterministic)."""
    carriers = [
        Carrier(c.carrier_id, c.carrier_type, c.lang, c.hard_negative, c.generator, text=c.text)
        for c in carriers_llm.load()
    ]
    carriers += [
        Carrier(f.carrier_id, f.carrier_type, f.lang, False, "faker", faker=f)
        for f in generate_faker_carriers(faker_counts)
    ]
    return carriers


class GoalFactory:
    """Instantiates training goals with random, non-canary slot values."""

    def __init__(self, seed: int) -> None:
        self._fakers = {"en": Faker("en_US"), "hu": Faker("hu_HU")}
        for index, fake in enumerate(self._fakers.values()):
            fake.seed_instance(seed + index)
        self._file_faker = Faker("en_US")
        self._file_faker.seed_instance(seed + 99)

    def user_name(self, lang: str) -> str:
        """A random first name."""
        return self._fakers[lang].first_name()

    def goal(self, lang: str, rng: random.Random) -> str:
        """A random training goal with slots filled."""
        fake = self._fakers[lang]
        values = {
            "email": fake.email(),
            "iban": fake.iban(),
            "amount": str(rng.choice((250, 480, 900, 1200, 1500, 2500, 4999))),
            "url": f"https://{self._file_faker.domain_name()}/{self._file_faker.uri_path()}",
            "file": f"{rng.choice(_FILE_DIRS)}/{self._file_faker.word()}_{rng.randint(2019, 2026)}."
            f"{rng.choice(_FILE_EXTS)}",
            "product": f"{fake.company()} {rng.choice(_MODEL_SUFFIXES)}",
            "person": fake.name(),
            "address": fake.address().replace("\n", ", "),
        }
        return fill(rng.choice(TRAIN_GOALS[lang]), **values)


def make_payload(template: Template, goal: str, user_name: str, rng: random.Random) -> str:
    """Instantiate a template; U2 templates obfuscate a plain instruction first."""
    if template.family == "U2":
        transform = template.id.split("-")[-1]
        base = fill(rng.choice(U2_BASE[template.lang]), goal=goal)
        return fill(template.text, payload=obfuscate(transform, base, rng, template.lang))
    return fill(template.text, goal=goal, user_name=user_name)


def _contains_canary(text: str) -> bool:
    lowered = text.lower()
    return (
        ATTACKER_DOMAIN in lowered
        or ATTACKER_EMAIL in lowered
        or MISINFO_CODE.lower() in lowered
        or normalize_iban(ATTACKER_IBAN) in normalize_iban(text)
    )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class DatasetBuilder:
    """Builds all samples; ``length_fn`` counts model tokens (injectable for tests)."""

    def __init__(self, length_fn: Callable[[str], int], seed: int = 42, max_tokens: int = MAX_TOKENS) -> None:
        self.length_fn = length_fn
        self.max_tokens = max_tokens
        self.rng = random.Random(seed)
        self.goals = GoalFactory(seed)
        self.templates = all_templates()
        pools = template_pools(self.templates)
        self.by_pool: dict[tuple[str, str, str], list[Template]] = collections.defaultdict(list)
        for template in self.templates:
            self.by_pool[(pools[template.id], template.family, template.lang)].append(template)
        self.samples: list[Sample] = []
        self._seen_texts: set[str] = set()
        self.dropped: collections.Counter[str] = collections.Counter()

    def _fit_carrier(self, text: str, payload_tokens: int) -> str:
        """Truncate a free-text carrier (word level) so carrier + payload fit in the token budget."""
        budget = self.max_tokens - payload_tokens - 12
        if self.length_fn(text) <= budget:
            return text
        words = text.split(" ")
        low, high = 1, len(words)
        while low < high:
            mid = (low + high + 1) // 2
            if self.length_fn(" ".join(words[:mid])) <= budget:
                low = mid
            else:
                high = mid - 1
        return " ".join(words[:low])

    def _add(self, sample: Sample) -> None:
        key = _normalise(sample.text)
        if key in self._seen_texts:
            self.dropped["duplicate"] += 1
            return
        if _contains_canary(sample.text):
            self.dropped["canary"] += 1
            return
        if self.length_fn(sample.text) > self.max_tokens:
            self.dropped["too_long"] += 1
            return
        self._seen_texts.add(key)
        self.samples.append(sample)

    def _pick_template(self, split: str, family: str, lang: str) -> Template:
        pool = "test" if family in UNSEEN_FAMILIES else split
        return self.rng.choice(self.by_pool[(pool, family, lang)])

    def _payload_lang(self, carrier_lang: str) -> str:
        if self.rng.random() < 0.1:
            return "hu" if carrier_lang == "en" else "en"
        return carrier_lang

    def add_benign(self, carrier: Carrier, split: str) -> None:
        """The carrier itself, label 0."""
        text = carrier.benign_text if carrier.faker else self._fit_carrier(carrier.text, 0)
        self._add(
            Sample(
                id=f"{carrier.carrier_id}-benign",
                text=text,
                label=0,
                lang=carrier.lang,
                carrier_type=carrier.carrier_type,
                carrier_id=carrier.carrier_id,
                split=split,
                hard_negative=carrier.hard_negative,
                generator=carrier.generator,
            )
        )

    def add_injected(self, carrier: Carrier, split: str, family: str, tag: str) -> None:
        """The carrier with one payload of ``family``, label 1."""
        lang = self._payload_lang(carrier.lang)
        template = self._pick_template(split, family, lang)
        payload = make_payload(template, self.goals.goal(lang, self.rng), self.goals.user_name(lang), self.rng)
        if carrier.faker:
            text, position, wrapper = carrier.faker.fill_slot(payload.strip()), "field", "json_field"
        else:
            position, wrapper = choose_position_and_wrapper(self.rng, family)
            host = self._fit_carrier(carrier.text, self.length_fn(payload))
            text = embed(host, payload, position, wrapper, self.rng)
        self._add(
            Sample(
                id=f"{carrier.carrier_id}-{tag}",
                text=text,
                label=1,
                lang=lang,
                family=family,
                carrier_type=carrier.carrier_type,
                position=position,
                carrier_id=carrier.carrier_id,
                template_id=template.id,
                split=split,
                hard_negative=carrier.hard_negative,
                generator=f"template+{carrier.generator}",
            )
        )

    def add_standalone(self, split: str, family: str, index: int) -> None:
        """A payload without carrier (direct-injection style), label 1."""
        lang = self.rng.choice(("en", "en", "hu"))
        template = self._pick_template(split, family, lang)
        payload = make_payload(template, self.goals.goal(lang, self.rng), self.goals.user_name(lang), self.rng)
        self._add(
            Sample(
                id=f"standalone-{split}-{family}-{index}",
                text=payload.strip(),
                label=1,
                lang=lang,
                family=family,
                position="standalone",
                template_id=template.id,
                split=split,
                generator="template",
            )
        )

    def add_snippet(self, carrier: Carrier, split: str) -> None:
        """A short benign excerpt (first sentences) of a free-text carrier, label 0.

        Balances the standalone payloads so that "short text" is not a shortcut feature for "injection".
        """
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", carrier.text.replace("\n", " ")) if part.strip()]
        snippet = " ".join(sentences[: self.rng.randint(1, 2)]).strip()
        if len(snippet.split()) < 4:
            return
        self._add(
            Sample(
                id=f"{carrier.carrier_id}-snippet",
                text=snippet,
                label=0,
                lang=carrier.lang,
                carrier_type="snippet",
                carrier_id=carrier.carrier_id,
                split=split,
                hard_negative=carrier.hard_negative,
                generator=carrier.generator,
            )
        )

    def build(self, carriers: list[Carrier], standalone: dict[str, int]) -> list[Sample]:
        """Benign + one seen-family injection per carrier; test carriers also get one unseen-family injection.

        Each split also gets ``standalone[split]`` bare payloads and as many short benign snippets.
        """
        by_split: dict[str, list[Carrier]] = collections.defaultdict(list)
        for index, carrier in enumerate(sorted(carriers, key=lambda c: c.carrier_id)):
            split = carrier_pool(carrier.carrier_id)
            by_split[split].append(carrier)
            self.add_benign(carrier, split)
            self.add_injected(carrier, split, SEEN[index % len(SEEN)], "seen")
            if split == "test":
                self.add_injected(carrier, split, UNSEEN_FAMILIES[index % len(UNSEEN_FAMILIES)], "unseen")
        for split, count in standalone.items():
            families = SEEN + (UNSEEN_FAMILIES if split == "test" else ())
            for index in range(count):
                self.add_standalone(split, families[index % len(families)], index)
            text_carriers = [c for c in by_split[split] if c.faker is None]
            for carrier in self.rng.sample(text_carriers, min(count, len(text_carriers))):
                self.add_snippet(carrier, split)
        logger.info("Built %d samples; dropped %s", len(self.samples), dict(self.dropped))
        return self.samples


def write_dataset(samples: list[Sample], out_dir: Path = DATASET_DIR) -> dict[str, object]:
    """Write one JSONL per split plus stats.json; return the stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, object] = {}
    for split in SPLITS:
        rows = sorted((s for s in samples if s.split == split), key=lambda s: s.id)
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in rows:
                handle.write(sample.model_dump_json() + "\n")
        stats[split] = {
            "total": len(rows),
            "by_label": dict(collections.Counter(s.label for s in rows)),
            "by_lang_label": {
                f"{k[0]}/{k[1]}": v for k, v in collections.Counter((s.lang, s.label) for s in rows).items()
            },
            "by_family": dict(sorted(collections.Counter(s.family for s in rows).items())),
            "hard_negative_benign": sum(1 for s in rows if s.hard_negative and s.label == 0),
            "by_carrier_type": dict(sorted(collections.Counter(s.carrier_type for s in rows).items())),
        }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote dataset to %s: %s", out_dir, {split: stats[split]["total"] for split in SPLITS})
    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--faker-en", type=int, default=900)
    parser.add_argument("--faker-hu", type=int, default=600)
    args = parser.parse_args()
    setup_logging()

    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def length_fn(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    carriers = load_carriers({"en": args.faker_en, "hu": args.faker_hu})
    logger.info("Loaded %d carriers", len(carriers))
    builder = DatasetBuilder(length_fn, seed=args.seed)
    samples = builder.build(carriers, standalone={"train": 300, "val": 60, "test": 120})
    write_dataset(samples)


if __name__ == "__main__":
    main()
