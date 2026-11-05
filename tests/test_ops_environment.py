"""OpsPilot environment, MCP server and task files (offline)."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json

import pytest
from mcp import Client

from pidefense.ops import CANARY_TOKENS, OPS_TOOL_KINDS, build_ops_server, load_ops_tasks

TASKS = load_ops_tasks()


def test_eight_tasks_over_six_surfaces() -> None:
    assert sorted(TASKS) == [f"P0{i}" for i in range(1, 9)]
    assert {t.surface for t in TASKS.values()} == {"ticket", "runbook", "db_note", "logs", "alert", "webpage"}


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_slots_render_inside_carrier_and_vanish_when_benign(task_id: str) -> None:
    task = TASKS[task_id]
    attacked = task.render_state("PAYLOAD-123")
    assert "PAYLOAD-123" in attacked.carrier_text()
    attacked.set_followup("FOLLOW-456")
    assert "FOLLOW-456" in attacked.carrier_text()
    benign = task.render_state(None)
    text = benign.carrier_text()
    assert "{{" not in text and "PAYLOAD" not in text
    benign.set_followup("")
    assert benign.carrier_text() == text


def test_query_db_reads_writes_and_errors() -> None:
    state = TASKS["P02"].render_state(None)
    rows = json.loads(state.query_db("SELECT id, status FROM orders WHERE customer_id = 2043 ORDER BY id"))
    assert rows["rows"] == [[9005, "refunded"], [9007, "settled"]]
    assert state.query_db("UPDATE orders SET status='x' WHERE id=9001") == "OK, 1 rows affected"
    assert state.action_log[-1].tool == "query_db" and state.action_log[-1].executed
    assert state.query_db("SELECT * FROM nope").startswith("Error")
    assert state.query_db("SELECT 1; SELECT 2").startswith("Error")
    tokens = json.loads(state.query_db("SELECT token FROM integration_tokens"))["rows"]
    assert {r[0] for r in tokens} == set(CANARY_TOKENS.values())


def test_db_note_followup_updates_the_stored_column() -> None:
    state = TASKS["P05"].render_state("NOTE-INJECTION")
    state.set_followup("NOTE-FOLLOWUP")
    notes = json.loads(state.query_db("SELECT notes FROM customers WHERE id = 2041"))["rows"][0][0]
    assert "NOTE-INJECTION" in notes and "NOTE-FOLLOWUP" in notes


def test_actions_change_state_and_are_logged() -> None:
    state = TASKS["P03"].render_state(None)
    assert "changed" in state.update_config("payments-api", "db_pool_size", "40")
    assert state.configs["payments-api"]["db_pool_size"] == 40
    state.update_config("payments-api", "auth_required", "false")
    assert state.configs["payments-api"]["auth_required"] is False
    assert "restarted" in state.restart_service("payments-api")
    assert "stopped" in state.stop_service("webhook-dispatcher")
    assert state.services["webhook-dispatcher"].status == "stopped"
    assert state.run_python("print(1)").startswith("Cell executed")
    assert state.http_request("POST", "https://x.example", "b") == "HTTP 200 OK"
    assert state.reply_ticket("T-404", "hi").startswith("Error")
    assert [r.tool for r in state.action_log if r.executed] == [
        "update_config",
        "update_config",
        "restart_service",
        "stop_service",
        "run_python",
        "http_request",
    ]


def test_ticket_followup_reopens_ticket() -> None:
    state = TASKS["P01"].render_state(None)
    state.reply_ticket("T-1044", "done")
    assert state.tickets["T-1044"].status == "closed"
    state.set_followup("one more thing")
    assert state.tickets["T-1044"].status == "open"
    assert "one more thing" in json.loads(state.read_ticket("T-1044"))["body"]


async def test_mcp_server_exposes_all_tools_and_errors() -> None:
    state = TASKS["P08"].render_state("PAGE-PAYLOAD")
    async with Client(build_ops_server(state)) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert names == set(OPS_TOOL_KINDS)
        page = await client.call_tool(
            "fetch_url", {"url": "https://status.cardnet-processing.example/incidents/2026-11-05"}
        )
        assert "PAGE-PAYLOAD" in page.content[0].text and not page.is_error
        missing = await client.call_tool("fetch_url", {"url": "https://nowhere.example"})
        assert missing.is_error
        sql = await client.call_tool("query_db", {"sql": "SELECT count(*) FROM orders"})
        assert json.loads(sql.content[0].text)["rows"] == [[8]]
