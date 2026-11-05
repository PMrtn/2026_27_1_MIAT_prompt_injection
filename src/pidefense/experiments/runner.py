"""Run the end-to-end experiment: 12 tasks x (16 attacks + 3 benign repetitions) x 4 defense configurations.

Configurations run in the order none -> guardrails -> ml -> combined. Every LLM request goes through an
exact-request cache: when a defense leaves a tool output unchanged, the later configurations send the same request
as configuration 1 and reuse its response (common random numbers). Episodes are paired across configurations,
and outcomes differ only where a defense intervened. Benign repetitions use distinct cache namespaces so they are
independent samples.

Usage:
    python -m pidefense.experiments.runner --grid-only            # write the attack grid
    python -m pidefense.experiments.runner --pilot 32             # config 'none' on a stratified sample
    python -m pidefense.experiments.runner                        # full run (resumable)
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

from pidefense.agent import run_episode
from pidefense.config import REPO_ROOT, LLMSettings
from pidefense.defenses import DefenseConfig, DefensePipeline
from pidefense.defenses.guardrails import build_guardrails
from pidefense.experiments.attacks import AttackSpec, build_attack_grid, load_grid, save_grid
from pidefense.experiments.scenarios import Scenario, load_scenarios
from pidefense.llm import AzureChatClient, BudgetExceededError, ResponseCache
from pidefense.logging_utils import setup_logging
from pidefense.schemas import EpisodeResult

logger = logging.getLogger(__name__)

E2E_DIR = REPO_ROOT / "results" / "e2e"
GRID_PATH = E2E_DIR / "attack_grid.jsonl"
CACHE_PATH = E2E_DIR / "llm_cache.jsonl"
BENIGN_REPS = 3


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode to run."""

    episode_id: str
    config: DefenseConfig
    task_id: str
    attack: AttackSpec | None
    repetition: int

    @property
    def cache_namespace(self) -> str:
        """Shared across configurations (pairing); distinct per benign repetition."""
        return "attack" if self.attack else f"benign-rep{self.repetition}"


def plan_episodes(
    scenarios: dict[str, Scenario], grid: list[AttackSpec], configs: list[DefenseConfig], benign_reps: int
) -> list[EpisodeSpec]:
    """Benign repetitions and every attack of the grid, for each configuration (in configuration order)."""
    specs: list[EpisodeSpec] = []
    for config in configs:
        for task_id in sorted(scenarios):
            for rep in range(benign_reps):
                specs.append(EpisodeSpec(f"{config.name}|{task_id}|benign|{rep}", config, task_id, None, rep))
        for attack in grid:
            specs.append(EpisodeSpec(f"{config.name}|{attack.attack_id}", config, attack.task_id, attack, 0))
    return specs


def pilot_specs(grid: list[AttackSpec], scenarios: dict[str, Scenario], size: int, seed: int = 7) -> list[EpisodeSpec]:
    """Config 'none' only: a family-stratified sample of attacks plus one benign run per task."""
    rng = random.Random(seed)
    by_family: dict[str, list[AttackSpec]] = {}
    for attack in grid:
        by_family.setdefault(attack.family, []).append(attack)
    chosen: list[AttackSpec] = []
    while len(chosen) < size and any(by_family.values()):
        for family in sorted(by_family):
            if by_family[family] and len(chosen) < size:
                chosen.append(by_family[family].pop(rng.randrange(len(by_family[family]))))
    none = DefenseConfig(guardrails=False, ml=False)
    specs = [EpisodeSpec(f"none|{task_id}|benign|0", none, task_id, None, 0) for task_id in sorted(scenarios)]
    specs += [EpisodeSpec(f"none|{a.attack_id}", none, a.task_id, a, 0) for a in chosen]
    return specs


def build_pipelines(configs: list[DefenseConfig]) -> dict[str, DefensePipeline]:
    """One pipeline per configuration; the detector is loaded once and shared."""
    detector = None
    if any(config.ml for config in configs):
        from pidefense.detector.inference import (  # pylint: disable=import-outside-toplevel
            InjectionDetector,
            MLDetectorDefense,
        )

        detector = MLDetectorDefense(InjectionDetector())
    pipelines: dict[str, DefensePipeline] = {}
    for config in configs:
        outputs, policies = [], []
        if config.guardrails:
            guard_outputs, guard_policies = build_guardrails()
            outputs += guard_outputs
            policies += guard_policies
        if config.ml and detector is not None:
            outputs.append(detector)
        pipelines[config.name] = DefensePipeline(outputs, policies)
        logger.info("Config %s: layers %s", config.name, pipelines[config.name].layer_names)
    return pipelines


def _done_ids(path: Path) -> set[str]:
    """Episodes already completed without error (errored ones are re-run; the last record wins in analysis)."""
    if not path.exists():
        return set()
    status: dict[str, bool] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                status[record["episode_id"]] = record.get("error") is None
    return {episode_id for episode_id, ok in status.items() if ok}


# Non-reasoning deployments reject `reasoning_effort`; None removes the agent default from the request.
MODEL_LLM_PARAMS: dict[str, dict[str, object]] = {"gpt-4.1": {"reasoning_effort": None}}


async def run_specs(specs: list[EpisodeSpec], scenarios: dict[str, Scenario], out_path: Path, concurrency: int) -> None:
    """Run episodes config by config (so config 1 fills the cache first), appending results as JSONL."""
    done = _done_ids(out_path)
    todo = [spec for spec in specs if spec.episode_id not in done]
    logger.info("%d episodes planned, %d already done, %d to run", len(specs), len(done), len(todo))
    if not todo:
        return
    settings = LLMSettings.from_env()
    client = AzureChatClient(settings=settings, cache=ResponseCache(CACHE_PATH))
    pipelines = build_pipelines(sorted({spec.config for spec in todo}, key=lambda c: c.number))
    lock = asyncio.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_cost = client.ledger.total_usd

    async def one(spec: EpisodeSpec) -> None:
        attack = spec.attack
        try:
            result: EpisodeResult = await run_episode(
                client,
                scenarios[spec.task_id],
                attack.payload if attack else None,
                pipelines[spec.config.name],
                config_name=spec.config.name,
                model=settings.deployment,
                episode_id=spec.episode_id,
                attack_id=attack.attack_id if attack else None,
                family=attack.family if attack else None,
                lang=attack.lang if attack else None,
                goal=attack.goal if attack else None,
                repetition=spec.repetition,
                cache_namespace=spec.cache_namespace,
                llm_params=MODEL_LLM_PARAMS.get(settings.deployment),
            )
        except BudgetExceededError:
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Episode %s failed", spec.episode_id)
            result = EpisodeResult(
                episode_id=spec.episode_id,
                model=settings.deployment,
                config=spec.config.name,
                task_id=spec.task_id,
                attack_id=attack.attack_id if attack else None,
                family=attack.family if attack else None,
                lang=attack.lang if attack else None,
                goal=attack.goal if attack else None,
                repetition=spec.repetition,
                error=f"{type(exc).__name__}: {exc}",
            )
        async with lock:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(result.model_dump_json() + "\n")

    try:
        for config in sorted({spec.config for spec in todo}, key=lambda c: c.number):
            batch = [spec for spec in todo if spec.config == config]
            started = time.perf_counter()
            semaphore = asyncio.Semaphore(concurrency)

            async def guarded(spec: EpisodeSpec, sem: asyncio.Semaphore = semaphore) -> None:
                async with sem:
                    await one(spec)

            await asyncio.gather(*(guarded(spec) for spec in batch))
            logger.info(
                "Config %s: %d episodes in %.1f min; spent so far this run $%.4f",
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
    parser.add_argument("--grid-only", action="store_true", help="only (re)write the attack grid")
    parser.add_argument("--pilot", type=int, default=0, help="run config 'none' on N stratified attacks")
    parser.add_argument("--configs", default="none,guardrails,ml,combined")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    setup_logging()

    scenarios = load_scenarios()
    if args.grid_only or not GRID_PATH.exists():
        save_grid(build_attack_grid(scenarios), GRID_PATH)
        logger.info("Wrote %s", GRID_PATH)
        if args.grid_only:
            return
    grid = load_grid(GRID_PATH)
    if args.pilot:
        specs = pilot_specs(grid, scenarios, args.pilot)
        out = args.out or E2E_DIR / "pilot_episodes.jsonl"
    else:
        configs = [DefenseConfig.from_name(name) for name in args.configs.split(",")]
        configs.sort(key=lambda c: c.number)
        specs = plan_episodes(scenarios, grid, configs, BENIGN_REPS)
        out = args.out or E2E_DIR / "episodes.jsonl"
    asyncio.run(run_specs(specs, scenarios, out, args.concurrency))


if __name__ == "__main__":
    main()
