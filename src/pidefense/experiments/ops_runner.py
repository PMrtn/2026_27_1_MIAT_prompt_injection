"""Run the OpsPilot (high-privilege) experiment: 8 tasks x 24 attacks x prompt variants A/B x 4 configurations.

Every attack episode is one conversation with two levels: L0 (the injected carrier) and L1 (an escalation
follow-up in the same carrier claiming authorisation). Benign episodes run the same two levels with empty payloads.
Configurations run in the order none -> guardrails -> ml -> combined; requests are cached per prompt variant, so
configurations are paired (common random numbers). An optional side run adds level L2 (the engineer in the chat
authorises the request) on a subset, to test how far prompt rules hold against operator pressure.

To limit LLM calls, the main runs use a seeded stratified subset (``results/ops/subset.json``, 3 attacks per
family); the full attack set is evaluated offline by ``pidefense.experiments.layer_eval``. With ``--sample`` /
``--verify`` an episode stops after the first level at which the attack succeeded (L1 = retry after a refusal).

Usage:
    python -m pidefense.experiments.ops_runner --attacks-only          # write results/ops/attacks.jsonl
    python -m pidefense.experiments.ops_runner --subset-only           # write results/ops/subset.json
    python -m pidefense.experiments.ops_runner --sample                # config none, variants A+B, subset (+benign)
    python -m pidefense.experiments.ops_runner --verify 16 --configs guardrails,ml,combined   # variant B check
    python -m pidefense.experiments.ops_runner --pilot 32               # config none, variant B, sample
    python -m pidefense.experiments.ops_runner                          # full factorial run (not used: too costly)
    python -m pidefense.experiments.ops_runner --l2 48                  # side run: L0-L1-L2, config none
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from pidefense.config import REPO_ROOT, LLMSettings
from pidefense.defenses import DefenseConfig, DefensePipeline
from pidefense.defenses.guardrails import Sanitizer, SignatureFilter
from pidefense.defenses.ops_policy import ActionPolicyOps
from pidefense.experiments.ops_attacks import (
    OpsAttack,
    build_ops_attacks,
    load_ops_attacks,
    save_ops_attacks,
    select_subset,
)
from pidefense.llm import AzureChatClient, BudgetExceededError, ResponseCache
from pidefense.logging_utils import setup_logging
from pidefense.ops import OpsTask, load_ops_tasks, run_ops_episode
from pidefense.schemas import EpisodeResult

logger = logging.getLogger(__name__)

OPS_DIR = REPO_ROOT / "results" / "ops"
ATTACKS_PATH = OPS_DIR / "attacks.jsonl"
SUBSET_PATH = OPS_DIR / "subset.json"
CACHE_PATH = OPS_DIR / "llm_cache.jsonl"
MAX_STEPS_PER_LEVEL = 8
BENIGN_REPS = 3
VARIANTS = ("A", "B")


@dataclass(frozen=True)
class OpsSpec:
    """One ops episode to run."""

    episode_id: str
    variant: str
    config: DefenseConfig
    task_id: str
    attack: OpsAttack | None
    repetition: int
    levels: tuple[str, ...] = ("L0", "L1")

    @property
    def cache_namespace(self) -> str:
        """Shared across configurations within a prompt variant; distinct per benign repetition."""
        kind = "attack" if self.attack else f"benign-rep{self.repetition}"
        return f"ops-{self.variant}-{kind}-{'-'.join(self.levels)}"


def plan(tasks: dict[str, OpsTask], attacks: list[OpsAttack], configs: list[DefenseConfig]) -> list[OpsSpec]:
    """Full factorial plan in configuration order."""
    specs: list[OpsSpec] = []
    for config in configs:
        for variant in VARIANTS:
            for task_id in sorted(tasks):
                for rep in range(BENIGN_REPS):
                    specs.append(
                        OpsSpec(f"{variant}|{config.name}|{task_id}|benign|{rep}", variant, config, task_id, None, rep)
                    )
            for attack in attacks:
                specs.append(
                    OpsSpec(f"{variant}|{config.name}|{attack.attack_id}", variant, config, attack.task_id, attack, 0)
                )
    return specs


def _sample(attacks: list[OpsAttack], size: int, seed: int) -> list[OpsAttack]:
    rng = random.Random(seed)
    by_family: dict[str, list[OpsAttack]] = {}
    for attack in attacks:
        by_family.setdefault(attack.family, []).append(attack)
    chosen: list[OpsAttack] = []
    while len(chosen) < size and any(by_family.values()):
        for family in sorted(by_family):
            if by_family[family] and len(chosen) < size:
                chosen.append(by_family[family].pop(rng.randrange(len(by_family[family]))))
    return chosen


def subset_attacks(attacks: list[OpsAttack]) -> list[OpsAttack]:
    """Attacks listed in the pre-registered subset file, in file order."""
    ids = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))["attack_ids"]
    by_id = {a.attack_id: a for a in attacks}
    return [by_id[i] for i in ids]


def sample_plan(tasks: dict[str, OpsTask], subset: list[OpsAttack]) -> list[OpsSpec]:
    """Config none, both prompt variants: every subset attack plus one benign run per task."""
    none = DefenseConfig(guardrails=False, ml=False)
    specs: list[OpsSpec] = []
    for variant in VARIANTS:
        specs += [OpsSpec(f"{variant}|none|{t}|benign|0", variant, none, t, None, 0) for t in sorted(tasks)]
        specs += [OpsSpec(f"{variant}|none|{a.attack_id}", variant, none, a.task_id, a, 0) for a in subset]
    return specs


def verify_plan(
    tasks: dict[str, OpsTask], subset: list[OpsAttack], configs: list[DefenseConfig], size: int
) -> list[OpsSpec]:
    """Variant B, the first ``size`` subset attacks (one per family first) and the benign runs, per configuration."""
    by_family: dict[str, OpsAttack] = {}
    for attack in subset:
        by_family.setdefault(attack.family, attack)
    chosen = list(by_family.values()) + [a for a in subset if a not in by_family.values()]
    chosen = chosen[:size]
    specs: list[OpsSpec] = []
    for config in configs:
        specs += [OpsSpec(f"B|{config.name}|{t}|benign|0", "B", config, t, None, 0) for t in sorted(tasks)]
        specs += [OpsSpec(f"B|{config.name}|{a.attack_id}", "B", config, a.task_id, a, 0) for a in chosen]
    return specs


def pilot_plan(attacks: list[OpsAttack], size: int) -> list[OpsSpec]:
    """Config none, variant B, a family-stratified sample."""
    none = DefenseConfig(guardrails=False, ml=False)
    return [OpsSpec(f"pilot|B|none|{a.attack_id}", "B", none, a.task_id, a, 0) for a in _sample(attacks, size, 7)]


def l2_plan(attacks: list[OpsAttack], size: int) -> list[OpsSpec]:
    """Side run: config none, both variants, L0-L1-L2 on a family-stratified sample."""
    none = DefenseConfig(guardrails=False, ml=False)
    chosen = _sample(attacks, size, 11)
    return [
        OpsSpec(f"l2|{v}|none|{a.attack_id}", v, none, a.task_id, a, 0, ("L0", "L1", "L2"))
        for v in VARIANTS
        for a in chosen
    ]


class PipelineFactory:
    """Per-(config, task) pipelines; the detector is loaded once."""

    def __init__(self, configs: list[DefenseConfig]) -> None:
        self.detector = None
        if any(config.ml for config in configs):
            from pidefense.detector.inference import (  # pylint: disable=import-outside-toplevel
                InjectionDetector,
                MLDetectorDefense,
            )

            self.detector = MLDetectorDefense(InjectionDetector())

    def build(self, config: DefenseConfig, task: OpsTask) -> DefensePipeline:
        """Guardrails = R1 sanitizer + R2 signatures + ops permissions; ML = detector on read outputs."""
        outputs: list = []
        policies: list = []
        if config.guardrails:
            outputs += [Sanitizer(), SignatureFilter()]
            policies.append(ActionPolicyOps(task.policy))
        if config.ml and self.detector is not None:
            outputs.append(self.detector)
        return DefensePipeline(outputs, policies)


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    status: dict[str, bool] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                status[record["episode_id"]] = record.get("error") is None
    return {key for key, ok in status.items() if ok}


async def run(
    specs: list[OpsSpec], tasks: dict[str, OpsTask], out_path: Path, concurrency: int, stop_on_success: bool = False
) -> None:
    """Run specs config by config; append results as JSONL (resumable)."""
    done = _done(out_path)
    todo = [spec for spec in specs if spec.episode_id not in done]
    logger.info("%d ops episodes planned, %d done, %d to run", len(specs), len(done), len(todo))
    if not todo:
        return
    settings = LLMSettings.from_env()
    client = AzureChatClient(settings=settings, cache=ResponseCache(CACHE_PATH))
    configs = sorted({spec.config for spec in todo}, key=lambda c: c.number)
    factory = PipelineFactory(configs)
    lock = asyncio.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_cost = client.ledger.total_usd

    async def one(spec: OpsSpec) -> None:
        attack = spec.attack
        task = tasks[spec.task_id]
        try:
            result: EpisodeResult = await run_ops_episode(
                client,
                task,
                attack.payload if attack else "",
                attack.followup if attack else "",
                factory.build(spec.config, task),
                prompt_variant=spec.variant,
                config_name=spec.config.name,
                model=settings.deployment,
                episode_id=spec.episode_id,
                levels=spec.levels,
                attack_id=attack.attack_id if attack else None,
                family=attack.family if attack else None,
                lang=attack.lang if attack else None,
                goal=attack.goal if attack else None,
                repetition=spec.repetition,
                cache_namespace=spec.cache_namespace,
                max_steps_per_level=MAX_STEPS_PER_LEVEL,
                stop_on_success=stop_on_success,
            )
        except BudgetExceededError:
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Ops episode %s failed", spec.episode_id)
            result = EpisodeResult(
                episode_id=spec.episode_id,
                model=settings.deployment,
                config=spec.config.name,
                task_id=spec.task_id,
                domain="ops",
                attack_id=attack.attack_id if attack else None,
                family=attack.family if attack else None,
                lang=attack.lang if attack else None,
                goal=attack.goal if attack else None,
                repetition=spec.repetition,
                error=f"{type(exc).__name__}: {exc}",
            )
        record = json.loads(result.model_dump_json())
        record.setdefault("prompt_variant", spec.variant)
        async with lock:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        for config in configs:
            batch = [spec for spec in todo if spec.config == config]
            started = time.perf_counter()
            semaphore = asyncio.Semaphore(concurrency)

            async def guarded(spec: OpsSpec, sem: asyncio.Semaphore = semaphore) -> None:
                async with sem:
                    await one(spec)

            await asyncio.gather(*(guarded(spec) for spec in batch))
            logger.info(
                "Ops config %s: %d episodes in %.1f min; spent this run $%.4f",
                config.name,
                len(batch),
                (time.perf_counter() - started) / 60,
                client.ledger.total_usd - start_cost,
            )
    finally:
        await client.aclose()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attacks-only", action="store_true")
    parser.add_argument("--subset-only", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--verify", type=int, default=0)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--l2", type=int, default=0)
    parser.add_argument("--configs", default="none,guardrails,ml,combined")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    setup_logging()

    tasks = load_ops_tasks()
    if args.attacks_only or not ATTACKS_PATH.exists():
        save_ops_attacks(build_ops_attacks(tasks), ATTACKS_PATH)
        logger.info("Wrote %s", ATTACKS_PATH)
        if args.attacks_only:
            return
    attacks = load_ops_attacks(ATTACKS_PATH)
    if args.subset_only or not SUBSET_PATH.exists():
        ids = select_subset(attacks)
        SUBSET_PATH.write_text(
            json.dumps({"seed": 2027, "per_family": 3, "n": len(ids), "attack_ids": ids}, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote %s (%d attacks)", SUBSET_PATH, len(ids))
        if args.subset_only:
            return
    configs = sorted((DefenseConfig.from_name(n) for n in args.configs.split(",")), key=lambda c: c.number)
    stop_on_success = False
    if args.sample:
        specs, out, stop_on_success = (
            sample_plan(tasks, subset_attacks(attacks)),
            OPS_DIR / "sample_episodes.jsonl",
            True,
        )
    elif args.verify:
        specs = verify_plan(tasks, subset_attacks(attacks), [c for c in configs if c.name != "none"], args.verify)
        out, stop_on_success = OPS_DIR / "verify_episodes.jsonl", True
    elif args.pilot:
        specs, out = pilot_plan(attacks, args.pilot), OPS_DIR / "pilot_episodes.jsonl"
    elif args.l2:
        specs, out = l2_plan(attacks, args.l2), OPS_DIR / "l2_episodes.jsonl"
    else:
        specs, out = plan(tasks, attacks, configs), OPS_DIR / "episodes.jsonl"
    asyncio.run(run(specs, tasks, out, args.concurrency, stop_on_success))


if __name__ == "__main__":
    main()
