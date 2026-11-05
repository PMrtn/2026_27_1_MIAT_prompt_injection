"""Serve one scenario's simulated workspace over MCP stdio (demo / external MCP clients).

Usage:
    python -m pidefense.mcp_server --task T01 [--injection "..."]

Logs go to stderr; stdout carries the MCP protocol.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pidefense.experiments.scenarios import SCENARIO_DIR, load_scenarios
from pidefense.logging_utils import setup_logging
from pidefense.mcp_server.server import build_server

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Scenario id, e.g. T01")
    parser.add_argument("--injection", default=None, help="Injection text for the scenario's carrier")
    parser.add_argument("--scenarios", default=str(SCENARIO_DIR), help="Scenario directory")
    args = parser.parse_args()
    setup_logging()
    scenario = load_scenarios(Path(args.scenarios))[args.task]
    server = build_server(scenario.render_state(args.injection))
    logger.info("Serving scenario %s over stdio", scenario.id)
    server.run("stdio")


if __name__ == "__main__":
    main()
