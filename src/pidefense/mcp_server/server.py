"""MCP server exposing the simulated workspace as tools (official ``mcp`` SDK 2.x, ``MCPServer``)."""

from __future__ import annotations

import logging
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from pidefense.mcp_server.environment import WorkspaceState
from pidefense.schemas import ToolKind

logger = logging.getLogger(__name__)


def _checked(result: str) -> str:
    """Turn the environment's "Error..." strings into MCP tool errors (``is_error=True``, message preserved)."""
    if result.startswith("Error"):
        raise ToolError(result)
    return result


# Read tools return (possibly attacker-controlled) content and pass through the output defenses.
# Action tools change the world and pass through the action policy first.
TOOL_KINDS: dict[str, ToolKind] = {
    "list_emails": "read",
    "read_email": "read",
    "list_files": "read",
    "read_file": "read",
    "fetch_webpage": "read",
    "get_product_reviews": "read",
    "list_transactions": "read",
    "send_email": "action",
    "transfer_money": "action",
    "delete_file": "action",
}

SERVER_NAME = "sim-workspace"


def build_server(state: WorkspaceState) -> MCPServer:
    """Create an MCP server whose tools operate on ``state`` (one server per episode)."""
    server = MCPServer(
        SERVER_NAME,
        instructions="Personal workspace of the user: email, files, web, product reviews and bank account.",
        log_level="WARNING",
    )

    @server.tool(structured_output=False)
    def list_emails() -> str:
        """List the emails in the user's inbox with id, sender, subject, date and unread flag."""
        return _checked(state.list_emails())

    @server.tool(structured_output=False)
    def read_email(email_id: Annotated[str, Field(description="Id of the email, as shown by list_emails")]) -> str:
        """Read the full content of one email."""
        return _checked(state.read_email(email_id))

    @server.tool(structured_output=False)
    def list_files() -> str:
        """List the files in the user's cloud drive."""
        return _checked(state.list_files())

    @server.tool(structured_output=False)
    def read_file(path: Annotated[str, Field(description="File path, as shown by list_files")]) -> str:
        """Read the content of a file in the user's cloud drive."""
        return _checked(state.read_file(path))

    @server.tool(structured_output=False)
    def fetch_webpage(url: Annotated[str, Field(description="Full URL of the page")]) -> str:
        """Fetch a web page and return its text content."""
        return _checked(state.fetch_webpage(url))

    @server.tool(structured_output=False)
    def get_product_reviews(product: Annotated[str, Field(description="Product name")]) -> str:
        """Get customer reviews of a product from the online shop."""
        return _checked(state.get_product_reviews(product))

    @server.tool(structured_output=False)
    def list_transactions() -> str:
        """List recent transactions and the balance of the user's bank account."""
        return _checked(state.list_transactions())

    @server.tool(structured_output=False)
    def send_email(
        to: Annotated[str, Field(description="Recipient email address; separate several with commas")],
        subject: Annotated[str, Field(description="Subject line")],
        body: Annotated[str, Field(description="Plain-text message body")],
    ) -> str:
        """Send an email from the user's account."""
        return _checked(state.send_email(to, subject, body))

    @server.tool(structured_output=False)
    def transfer_money(
        iban: Annotated[str, Field(description="IBAN of the recipient account")],
        amount: Annotated[float, Field(description="Amount in EUR")],
        reference: Annotated[str, Field(description="Payment reference / description")] = "",
    ) -> str:
        """Transfer money from the user's bank account."""
        return _checked(state.transfer_money(iban, amount, reference))

    @server.tool(structured_output=False)
    def delete_file(path: Annotated[str, Field(description="File path, as shown by list_files")]) -> str:
        """Permanently delete a file from the user's cloud drive."""
        return _checked(state.delete_file(path))

    logger.debug("Built MCP server with %d tools", len(TOOL_KINDS))
    return server
