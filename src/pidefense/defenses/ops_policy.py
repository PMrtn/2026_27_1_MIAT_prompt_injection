"""R3-ops: generic least-privilege tool permissions for the OpsPilot domain (part of the guardrails layer).

The rules were written from a generic least-privilege specification only (no attack text was consulted), the
same way R1-R3 were written before any dataset template existed:

* ``query_db``        - no access to credential tables/columns; no DDL/DML unless a task allow-listed the statement.
* ``update_config``   - security keys are never changed; other keys only if the task allow-listed (service, key).
* ``restart_service`` - only services the task allow-listed; ``stop_service`` never on production.
* ``run_python``      - no network or shell use, no reads of /etc or .env files.
* ``http_request``    - only allow-listed hosts.
* ``reply_ticket``    - data-loss prevention: no secret-looking strings, no links outside allow-listed domains.

The policy object is built per task (``ActionPolicyOps(task.policy)``) and plugs into
``DefensePipeline.check_action``; every other tool is allowed.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from pidefense.defenses.base import ActionPolicy
from pidefense.schemas import DefenseDecision, ToolContext

logger = logging.getLogger(__name__)


class ConfigKey(BaseModel):
    """A (service, key) pair a task may legitimately change."""

    service: str
    key: str


class OpsPolicy(BaseModel):
    """Per-task permissions consumed by ``ActionPolicyOps``."""

    allowed_sql_writes: list[str] = Field(default_factory=list)  # regexes (re.search, case-insensitive)
    allowed_config_changes: list[ConfigKey] = Field(default_factory=list)
    allowed_service_ops: list[str] = Field(default_factory=list)  # services the task may restart
    prod_services: list[str] = Field(default_factory=list)
    allowed_hosts: list[str] = Field(default_factory=list)  # fnmatch patterns, e.g. "*.northwind.internal"
    allowed_link_domains: list[str] = Field(default_factory=list)  # replies may only link to these (and subdomains)


_CREDENTIAL = re.compile(r"(token|secret|password|passwd|api_?key|credential)", re.IGNORECASE)
_WRITE_FIRST = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "vacuum",
    "reindex",
}
_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_SQL_STRING = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_QUOTES = re.compile(r"[\"'`\[\]]")
_SECURITY_KEY = re.compile(r"^(auth|debug|tls|verify|ssl|cors|allowed_origins|rate_limit)", re.IGNORECASE)
_PY_FORBIDDEN = re.compile(
    r"\b(requests|urllib|urllib3|httpx|aiohttp|http\.client|socket|subprocess|ftplib|smtplib|paramiko)\b"
    r"|os\.system|os\.popen|os\.exec|\bcurl\b|\bwget\b|pip\s+install|/etc/|\.env\b",
    re.IGNORECASE,
)
_SECRET = re.compile(
    r"sk_(live|test)_[A-Za-z0-9_]+|xox[abpors]-[A-Za-z0-9-]+|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*KEY-----"
    r"|\b[0-9a-f]{32,}\b|[A-Za-z0-9+/_=-]{40,}"
)
_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)


def sql_is_write(sql: str) -> bool:
    """True for statements that change data or schema (single statement, comments and literals ignored)."""
    cleaned = _SQL_STRING.sub("''", _SQL_COMMENT.sub(" ", sql or ""))
    words = re.findall(r"[a-z_]+", cleaned.lower())
    if not words:
        return False
    first = words[0]
    if first in _WRITE_FIRST:
        return True
    if first == "pragma":
        return "=" in cleaned
    if first == "with":
        return bool(re.search(r"\b(insert|update|delete|replace)\b", cleaned, re.IGNORECASE))
    return False


def _host_allowed(url: str, patterns: list[str]) -> bool:
    host = (urlparse(url if "://" in url else f"https://{url}").hostname or "").lower()
    return bool(host) and any(fnmatch.fnmatch(host, pattern.lower()) for pattern in patterns)


def _link_allowed(url: str, domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in domains)


class ActionPolicyOps(ActionPolicy):
    """Least-privilege permissions of the ops tools (see module docstring)."""

    name = "r3_ops_permissions"

    def __init__(self, policy: OpsPolicy) -> None:
        self.policy = policy
        self._sql_allow = [re.compile(pattern, re.IGNORECASE) for pattern in policy.allowed_sql_writes]

    def _decision(self, allowed: bool, reason: str = "") -> DefenseDecision:
        return DefenseDecision(layer=self.name, allowed=allowed, reason=reason if not allowed else "")

    def check(self, ctx: ToolContext) -> DefenseDecision:
        args = ctx.arguments
        tool = ctx.tool_name
        if tool == "query_db":
            return self._check_sql(str(args.get("sql", "")))
        if tool == "update_config":
            return self._check_config(str(args.get("service", "")), str(args.get("key", "")))
        if tool == "restart_service":
            service = str(args.get("service", "")).strip()
            if service not in self.policy.allowed_service_ops:
                return self._decision(False, f"restarting '{service}' is not part of the approved task")
            return self._decision(True)
        if tool == "stop_service":
            service = str(args.get("service", "")).strip()
            if service in self.policy.prod_services or service not in self.policy.allowed_service_ops:
                return self._decision(False, f"stopping '{service}' is not permitted")
            return self._decision(True)
        if tool == "run_python":
            match = _PY_FORBIDDEN.search(str(args.get("code", "")))
            if match:
                return self._decision(
                    False, f"notebook code may not use network, shell or secret files ({match.group(0)!r})"
                )
            return self._decision(True)
        if tool == "http_request":
            url = str(args.get("url", ""))
            if not _host_allowed(url, self.policy.allowed_hosts):
                return self._decision(
                    False, f"outbound requests to '{urlparse(url).hostname or url}' are not permitted"
                )
            return self._decision(True)
        if tool == "reply_ticket":
            return self._check_reply(str(args.get("message", "")))
        return self._decision(True)

    def _check_sql(self, sql: str) -> DefenseDecision:
        # Checked on the RAW statement with quote characters removed: SQLite accepts quoted identifiers ("x", 'x' in
        # FROM position, `x`, [x]) and comment markers can sit inside literals, so stripping comments or literals
        # first would open bypasses. Mentioning a credential word anywhere (even in a literal) is a false positive
        # we accept.
        if _CREDENTIAL.search(_QUOTES.sub("", sql.lower())):
            return self._decision(False, "access to credential tables or columns is not permitted")
        if sql_is_write(sql) and not any(pattern.search(sql) for pattern in self._sql_allow):
            return self._decision(False, "statements that modify data or schema are not permitted for this task")
        return self._decision(True)

    def _check_config(self, service: str, key: str) -> DefenseDecision:
        if _SECURITY_KEY.match(key.strip()):
            return self._decision(False, f"security setting '{key}' may not be changed")
        allowed = any(c.service == service.strip() and c.key == key.strip() for c in self.policy.allowed_config_changes)
        if not allowed:
            return self._decision(False, f"changing '{service}.{key}' is not part of the approved task")
        return self._decision(True)

    def _check_reply(self, message: str) -> DefenseDecision:
        if _SECRET.search(message):
            return self._decision(False, "the reply contains a secret-looking string (data-loss prevention)")
        for url in _URL.findall(message):
            if not _link_allowed(url, self.policy.allowed_link_domains):
                return self._decision(False, f"the reply links to a non-approved domain ({urlparse(url).hostname})")
        return self._decision(True)
