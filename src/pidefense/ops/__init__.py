"""OpsPilot: a high-privilege operations agent domain (SQL, configs, services, notebook, tickets)."""

from pidefense.defenses.ops_policy import ActionPolicyOps, ConfigKey, OpsPolicy
from pidefense.ops.checkers import check_goal, check_utility, is_weakening
from pidefense.ops.environment import CANARY_TOKENS, FOLLOWUP, INJECTION, OpsEnvironment, OpsState, parse_carrier
from pidefense.ops.episode import DEFAULT_OPS_LLM_PARAMS, LEVELS, OpsEpisodeResult, run_ops_episode
from pidefense.ops.prompts import PERSONA, RULES, system_prompt
from pidefense.ops.server import OPS_TOOL_KINDS, POLICY_CHECKED_READS, build_ops_server
from pidefense.ops.tasks import OPS_GOALS, OpsTask, load_ops_tasks

__all__ = [
    "ActionPolicyOps",
    "CANARY_TOKENS",
    "ConfigKey",
    "DEFAULT_OPS_LLM_PARAMS",
    "FOLLOWUP",
    "INJECTION",
    "LEVELS",
    "OPS_GOALS",
    "OPS_TOOL_KINDS",
    "OpsEnvironment",
    "OpsEpisodeResult",
    "OpsPolicy",
    "OpsState",
    "OpsTask",
    "PERSONA",
    "POLICY_CHECKED_READS",
    "RULES",
    "build_ops_server",
    "check_goal",
    "check_utility",
    "is_weakening",
    "load_ops_tasks",
    "parse_carrier",
    "run_ops_episode",
    "system_prompt",
]
