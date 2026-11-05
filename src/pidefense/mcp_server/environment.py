"""Simulated personal workspace behind the MCP tools: emails, files, web pages, reviews and a bank account.

All state lives in memory and is created fresh for every episode. Read tools return realistic text or JSON;
action tools change the state and append an ``ActionRecord`` only when they actually take effect.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from pidefense.attacks.goals import normalize_iban

logger = logging.getLogger(__name__)


class Email(BaseModel):
    """One message in the inbox."""

    id: str
    sender: str
    sender_name: str = ""
    to: list[str] = Field(default_factory=lambda: ["alex.morgan@company.example"])
    subject: str
    date: str
    body: str
    unread: bool = True


class Review(BaseModel):
    """One customer review of a product."""

    author: str
    rating: int = Field(ge=1, le=5)
    date: str
    text: str


class Transaction(BaseModel):
    """One bank account movement (negative amount = outgoing)."""

    id: str
    date: str
    counterparty: str
    iban: str
    amount: float
    currency: str = "EUR"
    reference: str = ""
    status: str = "completed"


class Contact(BaseModel):
    """An address book entry; payees also carry an IBAN."""

    name: str
    email: str
    company: str = ""
    iban: str | None = None


class ActionRecord(BaseModel):
    """An action that took effect (sent email, executed transfer, deleted file)."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""


def split_recipients(to: str | list[str]) -> list[str]:
    """Split a comma/semicolon separated recipient string into normalised addresses."""
    items = to if isinstance(to, list) else re.split(r"[,;]", to or "")
    return [item.strip().lower() for item in items if item and item.strip()]


def normalize_path(path: str) -> str:
    """Strip whitespace, leading './' and '/' so equivalent file paths compare equal."""
    cleaned = (path or "").strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def normalize_url(url: str) -> str:
    """Lower-case, add a missing scheme and drop the trailing slash."""
    cleaned = (url or "").strip()
    if not re.match(r"^https?://", cleaned, flags=re.IGNORECASE):
        cleaned = "https://" + cleaned
    return cleaned.rstrip("/").lower()


class WorkspaceState(BaseModel):
    """The complete simulated environment of one episode."""

    owner_name: str = "Alex Morgan"
    owner_email: str = "alex.morgan@company.example"
    today: str = "2026-11-05"
    emails: list[Email] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
    webpages: dict[str, str] = Field(default_factory=dict)
    reviews: dict[str, list[Review]] = Field(default_factory=dict)
    transactions: list[Transaction] = Field(default_factory=list)
    contacts: list[Contact] = Field(default_factory=list)
    balance: float = 25000.0
    currency: str = "EUR"
    action_log: list[ActionRecord] = Field(default_factory=list)

    # ----------------------------------------------------------------- read tools
    def list_emails(self) -> str:
        """Inbox overview as JSON (newest first)."""
        rows = [
            {"id": e.id, "from": e.sender, "subject": e.subject, "date": e.date, "unread": e.unread}
            for e in sorted(self.emails, key=lambda e: e.date, reverse=True)
        ]
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def read_email(self, email_id: str) -> str:
        """Full email with headers; marks it as read."""
        for email in self.emails:
            if email.id == (email_id or "").strip():
                email.unread = False
                sender = f"{email.sender_name} <{email.sender}>" if email.sender_name else email.sender
                return (
                    f"From: {sender}\nTo: {', '.join(email.to)}\nDate: {email.date}\n"
                    f"Subject: {email.subject}\n\n{email.body}"
                )
        return f"Error: no email with id '{email_id}'. Use list_emails to see the available ids."

    def list_files(self) -> str:
        """All file paths with their size in characters, as JSON."""
        rows = [{"path": path, "size_chars": len(content)} for path, content in sorted(self.files.items())]
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def _resolve_file(self, path: str) -> str | None:
        wanted = normalize_path(path)
        if wanted in self.files:
            return wanted
        for existing in self.files:
            if existing.lower() == wanted.lower():
                return existing
        return None

    def read_file(self, path: str) -> str:
        """File content or an error."""
        resolved = self._resolve_file(path)
        if resolved is None:
            return f"Error: file not found: '{path}'. Use list_files to see the available files."
        return self.files[resolved]

    def fetch_webpage(self, url: str) -> str:
        """Page content (pre-rendered text) or a 404 error."""
        wanted = normalize_url(url)
        for page_url, content in self.webpages.items():
            if normalize_url(page_url) == wanted:
                return content
        return f"Error 404: page not found: {url}"

    def get_product_reviews(self, product: str) -> str:
        """Reviews of a product (case-insensitive, substring match on the product name)."""
        wanted = (product or "").strip().lower()
        match = next((name for name in self.reviews if name.lower() == wanted), None)
        if match is None:
            match = next((name for name in self.reviews if wanted and wanted in name.lower()), None)
        if match is None:
            known = ", ".join(sorted(self.reviews)) or "none"
            return f"Error: no reviews found for '{product}'. Products with reviews: {known}."
        reviews = self.reviews[match]
        average = sum(r.rating for r in reviews) / len(reviews) if reviews else 0.0
        lines = [f"Customer reviews for {match} ({len(reviews)} reviews, average rating {average:.1f}/5):"]
        lines += [f"- {r.rating}/5 by {r.author} on {r.date}: {r.text}" for r in reviews]
        return "\n".join(lines)

    def list_transactions(self) -> str:
        """Recent account movements as JSON (newest first), plus the balance."""
        rows = [t.model_dump() for t in sorted(self.transactions, key=lambda t: t.date, reverse=True)]
        return json.dumps(
            {"balance": round(self.balance, 2), "currency": self.currency, "transactions": rows},
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------------- action tools
    def send_email(self, to: str, subject: str, body: str) -> str:
        """Send an email (simulated)."""
        recipients = split_recipients(to)
        if not recipients or any("@" not in r for r in recipients):
            return f"Error: invalid recipient address(es): '{to}'."
        result = f"Email sent to {', '.join(recipients)} with subject '{subject}'."
        self.action_log.append(
            ActionRecord(
                tool="send_email", arguments={"to": recipients, "subject": subject, "body": body}, result=result
            )
        )
        logger.debug("send_email executed: %s", recipients)
        return result

    def transfer_money(self, iban: str, amount: float, reference: str = "") -> str:
        """Execute a bank transfer (simulated)."""
        normalized = normalize_iban(iban)
        if len(normalized) < 15 or not normalized[:2].isalpha():
            return f"Error: invalid IBAN '{iban}'."
        if amount <= 0:
            return "Error: the amount must be positive."
        if amount > self.balance:
            return f"Error: insufficient funds (balance {self.balance:.2f} {self.currency})."
        self.balance -= amount
        payee = next(
            (c.name for c in self.contacts if c.iban and normalize_iban(c.iban) == normalized), "Unknown payee"
        )
        transaction = Transaction(
            id=f"TX-{len(self.transactions) + 1:04d}",
            date=self.today,
            counterparty=payee,
            iban=iban,
            amount=-round(amount, 2),
            currency=self.currency,
            reference=reference,
        )
        self.transactions.append(transaction)
        result = (
            f"Transfer {transaction.id} executed: {amount:.2f} {self.currency} to {iban} ({payee}), "
            f"reference '{reference}'. New balance: {self.balance:.2f} {self.currency}."
        )
        self.action_log.append(
            ActionRecord(
                tool="transfer_money",
                arguments={"iban": iban, "amount": amount, "reference": reference},
                result=result,
            )
        )
        logger.debug("transfer_money executed: %s %.2f", normalized, amount)
        return result

    def delete_file(self, path: str) -> str:
        """Delete a file (simulated)."""
        resolved = self._resolve_file(path)
        if resolved is None:
            return f"Error: file not found: '{path}'."
        del self.files[resolved]
        result = f"File '{resolved}' deleted."
        self.action_log.append(ActionRecord(tool="delete_file", arguments={"path": resolved}, result=result))
        logger.debug("delete_file executed: %s", resolved)
        return result
