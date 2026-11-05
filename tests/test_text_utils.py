"""The shared email-address extraction used by the environment, the checkers and (later) R3."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import pytest

from pidefense.mcp_server import split_recipients
from pidefense.text_utils import extract_emails


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a@x.example", ["a@x.example"]),
        ("Attacker <Attacker@Evil.Example>", ["attacker@evil.example"]),
        ("mailto:anna.kovacs@company.example", ["anna.kovacs@company.example"]),
        ("bob@company.example and anna@company.example", ["bob@company.example", "anna@company.example"]),
        ("a@x.example, b@y.example; c@z.example", ["a@x.example", "b@y.example", "c@z.example"]),
        (["A <a@x.example>", ["b@y.example", "a@x.example"]], ["a@x.example", "b@y.example"]),
        ("Please write to bob@company.example.", ["bob@company.example"]),
        ("Bob", []),
        (None, []),
    ],
)
def test_extract_emails(value: object, expected: list[str]) -> None:
    assert extract_emails(value) == expected


def test_split_recipients_is_the_shared_extractor() -> None:
    assert split_recipients("Bob <bob@company.example>, mailto:x@evil.example") == extract_emails(
        "bob@company.example x@evil.example"
    )
