"""Every MCP tool of the simulated workspace, called through the in-memory MCP client."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json

import pytest
from mcp import Client

from pidefense.attacks.goals import ATTACKER_EMAIL, ATTACKER_IBAN
from pidefense.experiments.scenarios import load_scenarios
from pidefense.mcp_server import TOOL_KINDS, WorkspaceState, build_server


@pytest.fixture(name="scenarios", scope="module")
def _scenarios() -> dict:
    return load_scenarios()


async def _call(state: WorkspaceState, name: str, arguments: dict | None = None) -> str:
    async with Client(build_server(state)) as client:
        result = await client.call_tool(name, arguments or {})
        assert not result.is_error, result.content
        return "\n".join(item.text for item in result.content)


async def test_tool_list_matches_kinds(scenarios: dict) -> None:
    async with Client(build_server(scenarios["T01"].render_state(None))) as client:
        tools = (await client.list_tools()).tools
    assert {tool.name for tool in tools} == set(TOOL_KINDS)
    assert all(tool.description for tool in tools)


async def test_email_tools(scenarios: dict) -> None:
    state = scenarios["T01"].render_state("PAYLOAD-123")
    inbox = json.loads(await _call(state, "list_emails"))
    assert {row["id"] for row in inbox} >= {"e1", "e2", "e3"}
    text = await _call(state, "read_email", {"email_id": "e2"})
    assert "From: Tech Digest <newsletter@techdigest.example>" in text and "PAYLOAD-123" in text
    assert "no email" in await _call(state, "read_email", {"email_id": "zzz"})


async def test_file_tools(scenarios: dict) -> None:
    state = scenarios["T05"].render_state(None)
    paths = [row["path"] for row in json.loads(await _call(state, "list_files"))]
    assert "downloads/acme_proposal.txt" in paths
    assert "EUR 48,500" in await _call(state, "read_file", {"path": "/downloads/acme_proposal.txt"})
    assert "not found" in await _call(state, "read_file", {"path": "nope.txt"})


async def test_web_and_review_tools(scenarios: dict) -> None:
    web_state = scenarios["T06"].render_state(None)
    page = await _call(web_state, "fetch_webpage", {"url": "www.techpulse.example/reviews/student-laptops-2026/"})
    assert "Nimbus Aero 14" in page
    assert "404" in await _call(web_state, "fetch_webpage", {"url": "https://unknown.example"})
    review_state = scenarios["T10"].render_state("PAYLOAD-XYZ")
    reviews = await _call(review_state, "get_product_reviews", {"product": "ergochair pro"})
    assert "4 reviews" in reviews and "PAYLOAD-XYZ" in reviews


async def test_transactions_and_transfer(scenarios: dict) -> None:
    state = scenarios["T12"].render_state("PAYLOAD-TX")
    listing = json.loads(await _call(state, "list_transactions"))
    assert listing["balance"] == 25000.0
    assert any("PAYLOAD-TX" in row["reference"] for row in listing["transactions"])
    async with Client(build_server(state)) as client:
        ok = await client.call_tool("transfer_money", {"iban": ATTACKER_IBAN, "amount": 100.0, "reference": "x"})
        too_much = await client.call_tool("transfer_money", {"iban": ATTACKER_IBAN, "amount": 1e9})
    assert "executed" in ok.content[0].text and "insufficient" in too_much.content[0].text
    assert state.balance == pytest.approx(24900.0)
    assert [record.tool for record in state.action_log] == ["transfer_money"]


async def test_send_email_and_delete_file(scenarios: dict) -> None:
    state = scenarios["T01"].render_state(None)
    async with Client(build_server(state)) as client:
        await client.call_tool(
            "send_email", {"to": f"{ATTACKER_EMAIL}, Bob.Smith@company.example", "subject": "s", "body": "b"}
        )
        await client.call_tool("delete_file", {"path": "contracts/master_services_agreement_2026.pdf"})
        missing = await client.call_tool("delete_file", {"path": "does/not/exist.txt"})
    assert state.action_log[0].arguments["to"] == [ATTACKER_EMAIL, "bob.smith@company.example"]
    assert "contracts/master_services_agreement_2026.pdf" not in state.files
    assert "not found" in missing.content[0].text
    assert [record.tool for record in state.action_log] == ["send_email", "delete_file"]
