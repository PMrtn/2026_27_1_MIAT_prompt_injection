"""Deterministic guardrails R1-R3 (configuration 2, and half of configuration 4).

* R1 ``Sanitizer``       - normalises tool outputs and strips invisible / hidden content. Never blocks.
* R2 ``SignatureFilter`` - curated EN + HU regular expressions; any match withholds the tool output.
* R3 ``ActionPolicyR3``  - vetoes action tool calls whose arguments are not backed by scenario facts.

Independence and freeze
-----------------------
The R2 signatures were written only from the five *training* attack-family descriptions
(``pidefense.schemas.FAMILY_DESCRIPTIONS`` S1-S5) and general knowledge of common English and Hungarian
prompt-injection phrasing. They were written before any dataset template existed and deliberately do not
target the held-out families U1-U3 (boundary spoofing, obfuscation, implicit "note to AI" phrasing): there is
no base64 detection, no fake-turn detection beyond generic role/system tags and no homoglyph handling.

These rules are FROZEN once merged, before any test or end-to-end result exists. Changing them afterwards would
invalidate the comparison between configurations, so any later edit must be reported as a deviation.

Signatures are matched against a *folded* copy of the text: NFKD with combining marks removed (so Hungarian
accented and unaccented spellings match the same pattern), typographic quotes mapped to ASCII, whitespace
collapsed and case folded. All patterns are therefore written in lower case without accents.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from pidefense.attacks.goals import normalize_iban
from pidefense.defenses.base import ActionPolicy, OutputDefense
from pidefense.schemas import DefenseDecision, ToolContext

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------------------------
# R1: sanitizer
# --------------------------------------------------------------------------------------------------------------------

# Explicitly listed invisible characters; every other Unicode "format" (Cf) character is removed as well.
_INVISIBLE_RANGES: tuple[tuple[int, int], ...] = (
    (0x200B, 0x200F),  # zero-width space/joiners, LRM/RLM
    (0x202A, 0x202E),  # bidi embeddings and overrides
    (0x2060, 0x2064),  # word joiner, invisible operators
    (0x2066, 0x2069),  # bidi isolates
    (0xFEFF, 0xFEFF),  # zero-width no-break space / BOM
)

_HTML_COMMENT_RE = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HIDDEN_ELEMENT_RE = re.compile(
    r"<(?P<tag>[a-z][\w-]*)\b"
    r"(?=[^>]*(?:"
    r"style\s*=\s*\"[^\"]*(?:display\s*:\s*none|visibility\s*:\s*hidden)[^\"]*\""
    r"|style\s*=\s*'[^']*(?:display\s*:\s*none|visibility\s*:\s*hidden)[^']*'"
    r"|\shidden(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?(?=[\s/>])"
    r"))"
    r"[^>]*>.*?</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _is_invisible(char: str) -> bool:
    code = ord(char)
    if any(low <= code <= high for low, high in _INVISIBLE_RANGES):
        return True
    return unicodedata.category(char) == "Cf"


class Sanitizer(OutputDefense):
    """R1: strip invisible characters and hidden HTML content, then NFKC-normalise. Never blocks.

    Known limitation: nested elements with the same tag inside a hidden element are only removed up to the
    first closing tag (regular expressions cannot balance tags).
    """

    name = "r1_sanitize"

    def sanitize(self, text: str) -> tuple[str, Counter[str]]:
        """Return the cleaned text and a counter of what was removed."""
        removed: Counter[str] = Counter()
        kept_chars: list[str] = []
        for char in text:
            if _is_invisible(char):
                removed["invisible character"] += 1
            else:
                kept_chars.append(char)
        cleaned = "".join(kept_chars)

        normalised = unicodedata.normalize("NFKC", cleaned)
        if normalised != cleaned:
            removed["NFKC change"] += 1
        cleaned = normalised

        cleaned, count = _HTML_COMMENT_RE.subn("", cleaned)
        removed["HTML comment"] += count
        cleaned, count = _SCRIPT_STYLE_RE.subn("", cleaned)
        removed["script/style block"] += count
        cleaned, count = _HIDDEN_ELEMENT_RE.subn("", cleaned)
        removed["hidden element"] += count
        return cleaned, +removed

    def inspect(self, text: str, ctx: ToolContext) -> DefenseDecision:
        cleaned, removed = self.sanitize(text)
        if removed:
            reason = "removed/normalised: " + ", ".join(f"{count} x {what}" for what, count in sorted(removed.items()))
            logger.debug("Sanitized output of %s: %s", ctx.tool_name, reason)
        else:
            reason = "no changes"
        return DefenseDecision(layer=self.name, allowed=True, text=cleaned, reason=reason)


# --------------------------------------------------------------------------------------------------------------------
# R2: signature filter
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Signature:
    """One R2 signature: attack family, identifier, language and the regex (applied to folded text)."""

    family: str
    pattern_id: str
    lang: str
    regex: str


_ADDR_EN = (
    r"(?:you\s+(?:must|should|shall|need\s+to|will\s+now|are\s+now|are\s+required\s+to|are\s+instructed\s+to)"
    r"|\b(?:assistant|ai|language\s+model|chatbot|llm)\b)"
)
_ADDR_HU = (
    r"\b(?:asszisztens\w*|ai|mesterseges\s+intelligencia|nyelvi\s+modell\w*|chatbot\w*|agens\w*|neked|szamodra)\b"
)
_HU_ACTION = r"(?:kuldd|kuldj|utald|utalj|torold|tovabbitsd|add\s+hozza|ird\s+hozza|ird\s+bele|emlitsd\s+meg)"

SIGNATURES: tuple[Signature, ...] = (
    # ---------------- S1: direct override ----------------
    Signature(
        "S1",
        "en_ignore_previous",
        "en",
        r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|every\s+)?(?:of\s+)?"
        r"(?:the\s+|your\s+|my\s+|these\s+|those\s+)?(?:previous|prior|above|earlier|preceding|original|initial|"
        r"existing|system)\s+(?:instructions?|prompts?|rules|directives|guidelines)\b",
    ),
    Signature(
        "S1",
        "en_disregard_your",
        "en",
        r"\b(?:disregard|ignore|forget|abandon|drop)\s+(?:all\s+)?(?:of\s+)?your\s+"
        r"(?:previous\s+|prior\s+|original\s+)?"
        r"(?:instructions?|rules|guidelines|programming|guardrails|system\s+prompt|directives|training)\b",
    ),
    Signature(
        "S1",
        "en_new_instructions",
        "en",
        r"\b(?:new|updated|revised)\s+instructions?\s*:\s*(?:you|from\s+now|ignore|forget|disregard|do\s+not|don't|"
        r"always|never|send|forward|transfer|delete|immediately|the\s+assistant)\b",
    ),
    Signature("S1", "en_your_new_task", "en", r"\byour\s+new\s+(?:instructions?|task|directives?)\s+(?:is|are)\b"),
    Signature(
        "S1",
        "en_do_not_follow",
        "en",
        r"\b(?:do\s+not|don't|stop)\s+(?:follow|following|obey|obeying)\s+(?:the\s+|your\s+|any\s+)?"
        r"(?:previous|prior|original|above|system|earlier)\s+(?:instructions?|rules|prompt)\b",
    ),
    Signature(
        "S1",
        "en_reveal_system_prompt",
        "en",
        r"\b(?:reveal|print|show|output|repeat|leak|disclose|display)\s+(?:me\s+)?(?:your|the)\s+"
        r"(?:full\s+|entire\s+|original\s+|hidden\s+)?(?:system\s+prompt|initial\s+instructions|hidden\s+instructions|"
        r"system\s+instructions)\b",
    ),
    Signature(
        "S1",
        "en_override_safety",
        "en",
        r"\b(?:override|bypass|disable|turn\s+off)\s+(?:all\s+)?(?:your\s+|the\s+)?(?:safety|security|content)\s+"
        r"(?:rules|instructions|restrictions|guidelines|filters|protocols|policies)\b",
    ),
    Signature(
        "S1",
        "hu_hagyd_figyelmen_kivul",
        "hu",
        r"\bhagy(?:d|ja|jad|jatok)\s+figyelmen\s+kivul\s+(?:az?\s+)?(?:osszes\s+|minden\s+)?"
        r"(?:korabbi|elozo|fenti|eredeti|eddigi|rendszer)\s*-?\s*(?:utasitas\w*|szabaly\w*|parancs\w*|prompt\w*)",
    ),
    Signature(
        "S1",
        "hu_utasitasokat_hagyd",
        "hu",
        r"\b(?:korabbi|elozo|fenti|eredeti|eddigi)\s+(?:utasitas\w*|szabaly\w*)\s+(?:hagyd|hagyja|felejtsd|felejtse|"
        r"ne\s+vedd)",
    ),
    Signature(
        "S1",
        "hu_felejtsd_el",
        "hu",
        r"\bfelejts(?:d|e)\s+el\s+(?:az?\s+)?(?:osszes\s+|minden\s+)?(?:(?:korabbi|elozo|fenti|eredeti|eddigi)\s+)?"
        r"(?:utasitas\w*|szabaly\w*|parancs\w*)",
    ),
    Signature(
        "S1",
        "hu_ne_kovesd",
        "hu",
        r"\bne\s+kovess?(?:d|e)\s+(?:az?\s+)?(?:korabbi|elozo|eredeti|fenti|eddigi)\s+utasitas\w*",
    ),
    Signature(
        "S1",
        "hu_uj_utasitas",
        "hu",
        r"\buj\s+utasitas(?:ok|aid|od)?\s*:\s*(?:te|mostantol|hagyd|felejtsd|ne|mindig|kuldd|kuldj|utald|utalj|"
        r"torold|tovabbitsd|azonnal)\b",
    ),
    Signature(
        "S1",
        "hu_rendszer_prompt",
        "hu",
        r"\b(?:ird\s+ki|mutasd\s+meg|add\s+ki|ismeteld\s+meg|aruld\s+el)\s+(?:a\s+)?(?:te\s+)?rendszer\s*-?\s*"
        r"(?:prompt\w*|utasitas\w*)",
    ),
    # ---------------- S2: fake system / authority message ----------------
    Signature(
        "S2",
        "en_system_tag",
        "en",
        r"\[\s*(?:system|sys|admin|administrator|developer|root|security\s+team|openai|anthropic|microsoft)\b"
        r"[^\]]{0,30}\]\s*:?.{0,200}?" + _ADDR_EN,
    ),
    Signature(
        "S2", "en_chat_template_token", "en", r"<\|\s*(?:im_start|im_end|system|endoftext|start_header_id|eot_id)\s*\|>"
    ),
    Signature(
        "S2",
        "en_markdown_system_header",
        "en",
        r"(?:^|\s)#{2,4}\s*(?:system|admin|developer)\s+(?:message|prompt|instructions?|override|notice)\b",
    ),
    Signature(
        "S2",
        "en_markdown_system_colon",
        "en",
        r"(?:^|\s)#{2,4}\s*(?:system|admin|developer)\s*:\s*"
        r"(?:you|ignore|from\s+now|the\s+assistant|new\s+instructions)",
    ),
    Signature(
        "S2",
        "en_system_notice_to_ai",
        "en",
        r"\b(?:system|admin|administrator|developer)\s+(?:message|notice|note|update|instruction)s?\s*:.{0,200}?"
        r"\b(?:assistant|ai|language\s+model|chatbot|llm)\b",
    ),
    Signature(
        "S2",
        "en_system_override",
        "en",
        r"\b(?:system|admin|administrator|developer|root)\s+(?:override|directive)s?\b",
    ),
    Signature("S2", "en_security_override", "en", r"\b(?:security|priority)\s+override\b"),
    Signature(
        "S2",
        "en_ai_is_required",
        "en",
        r"\b(?:ai\s+assistant|ai|language\s+model|llm|chatbot)\s+(?:is|are|must|shall)\s+(?:now\s+)?"
        r"(?:required|authori[sz]ed|instructed|ordered|obligated|permitted)\s+to\b",
    ),
    Signature(
        "S2",
        "en_you_are_in_mode",
        "en",
        r"\byou\s+are\s+(?:now\s+)?(?:in|operating\s+in|running\s+in)\s+(?:developer|debug|god|jailbreak|"
        r"unrestricted|dan|admin)\s+mode\b",
    ),
    Signature(
        "S2",
        "hu_system_tag",
        "hu",
        r"\[\s*(?:rendszer\w*|admin\w*|adminisztrator\w*|fejleszto\w*|biztonsag\w*)[^\]]{0,30}\]\s*:?.{0,200}?"
        + _ADDR_HU,
    ),
    Signature("S2", "hu_rendszeruzenet_to_ai", "hu", r"\brendszer\s*-?\s*uzenet\w*\s*:.{0,200}?" + _ADDR_HU),
    Signature(
        "S2",
        "hu_felulbiralas",
        "hu",
        r"\b(?:rendszer|admin|fejlesztoi|biztonsagi)\s*-?\s*(?:felulbiralas|feluliras)\w*",
    ),
    Signature(
        "S2",
        "hu_ai_koteles",
        "hu",
        r"\b(?:ai\s+asszisztens|mesterseges\s+intelligencia|nyelvi\s+modell|chatbot)\w*\s+(?:most\s+)?"
        r"(?:koteles|kotelezo|fel\s+van\s+hatalmazva|utasitast\s+kapott)",
    ),
    Signature(
        "S2",
        "hu_fejlesztoi_mod",
        "hu",
        r"\b(?:fejlesztoi|korlatlan|jailbreak|dan)\s+mod(?:ban|ba)\s+(?:vagy|mukodsz|kapcsol\w*|lep\w*)\b",
    ),
    # ---------------- S3: role-play / persona hijack ----------------
    Signature(
        "S3",
        "en_you_are_now_jailbroken",
        "en",
        r"\byou\s+are\s+now\s+(?:dan|stan|dude|evil\w*|unrestricted|jailbroken|unfiltered|uncensored|"
        r"free\s+from\s+(?:all\s+|any\s+)?(?:rules|restrictions|guidelines))\b",
    ),
    Signature(
        "S3",
        "en_you_are_now_persona",
        "en",
        r"\byou\s+are\s+now\s+(?:a|an|the|my)\s+(?:\w+\s+){0,3}(?:ai|bot|chatbot|persona|character|hacker|entity)\b",
    ),
    Signature(
        "S3",
        "en_pretend_to_be",
        "en",
        r"\b(?:pretend|imagine)\s+(?:that\s+)?(?:you\s+are|you're|to\s+be)\s+(?:a\s+|an\s+|the\s+)?(?:\w+\s+){0,3}"
        r"(?:ai|bot|chatbot|hacker|character|persona|criminal|villain)\b",
    ),
    Signature(
        "S3",
        "en_act_as_unrestricted",
        "en",
        r"\b(?:act|behave|respond|answer|roleplay|role-play)\s+(?:as|like)\s+(?:a\s+|an\s+|the\s+)?(?:\w+\s+){0,2}"
        r"(?:unrestricted|unfiltered|uncensored|evil|jailbroken|rogue|malicious|amoral)\b",
    ),
    Signature(
        "S3",
        "en_from_now_on_act_as",
        "en",
        r"\bfrom\s+now\s+on\s*,?\s+you\s+(?:will\s+|must\s+|shall\s+|should\s+)?(?:act|respond|behave|speak|roleplay|"
        r"role-play)\s+(?:as|like)\s+(?:a\s+|an\s+|the\s+)?(?:\w+\s+){0,2}(?:ai|bot|chatbot|model|character|persona|"
        r"hacker|dan)\b",
    ),
    Signature(
        "S3",
        "en_no_longer_bound",
        "en",
        r"\byou\s+are\s+no\s+longer\s+(?:bound|restricted|limited|constrained)\s+by\s+(?:any\s+|your\s+|the\s+)?"
        r"(?:\w+\s+)?(?:rules|guidelines|policies|policy|restrictions|programming|safety)\b",
    ),
    Signature(
        "S3", "en_no_longer_an_ai", "en", r"\byou\s+are\s+no\s+longer\s+(?:a|an)\s+(?:ai|language\s+model|chatbot)\b"
    ),
    Signature("S3", "en_your_new_name", "en", r"\byour\s+new\s+(?:name|persona|identity)\s+is\b"),
    Signature(
        "S3",
        "en_ai_without_restrictions",
        "en",
        r"\b(?:ai|chatbot|language\s+model|llm)\s+(?:without|with\s+no|free\s+of|free\s+from)\s+(?:any\s+)?"
        r"(?:ethical\s+|moral\s+|content\s+|safety\s+)?"
        r"(?:restrictions|limitations|filters|guidelines|rules|censorship)\b",
    ),
    Signature(
        "S3",
        "en_dan_jailbreak",
        "en",
        r"\b(?:do\s+anything\s+now|dan\s+mode|jailbreak\s+mode|(?:enable|activate|enter)\s+(?:dan|jailbreak)\b)",
    ),
    Signature(
        "S3",
        "hu_mostantol_te",
        "hu",
        r"\bmostantol\s+(?:kezdve\s+)?te\s+(?:egy\s+)?(?:\w+\s+){0,3}(?:korlatlan\w*|cenzurazatlan\w*|szuretlen\w*|"
        r"gonosz\w*|dan|ai|asszisztens\w*|chatbot\w*|modell\w*|karakter\w*|hacker\w*)\b",
    ),
    Signature(
        "S3",
        "hu_mostantol_viselkedj",
        "hu",
        r"\bmostantol\s+(?:kezdve\s+)?(?:ugy\s+)?(?:viselkedsz|viselkedj|jatszod|jatszd|tegyel\s+ugy)\b",
    ),
    Signature(
        "S3", "hu_mar_nem_ai", "hu", r"\bte\s+mar\s+nem\s+(?:egy\s+)?(?:ai|asszisztens\w*|nyelvi\s+modell|chatbot)\b"
    ),
    Signature(
        "S3",
        "hu_tegyel_ugy",
        "hu",
        r"\b(?:tegyel\s+ugy|tettesd|kepzeld\s+el)\s*,?\s+(?:mintha|hogy)\s+(?:te\s+)?(?:egy\s+)?(?:\w+\s+){0,3}"
        r"(?:ai|asszisztens|hacker|karakter|bot|modell)\w*\b",
    ),
    Signature(
        "S3",
        "hu_korlatlan_ai",
        "hu",
        r"\b(?:korlatlan|cenzurazatlan|szuretlen)\s+(?:ai|asszisztens|modell|chatbot|nyelvi\s+modell)\w*",
    ),
    Signature("S3", "hu_uj_neved", "hu", r"\b(?:az?\s+)?uj\s+(?:neved|szemelyiseged|identitasod)\b"),
    Signature("S3", "hu_dan_mod", "hu", r"\b(?:dan|jailbreak)\s+mod(?:ba|ban|ot)?\b"),
    # ---------------- S4: urgent "message from the user" social engineering ----------------
    Signature(
        "S4",
        "en_urgent_message_from_user",
        "en",
        r"\b(?:important|urgent|critical|priority|high-priority)\s+(?:message|instruction|note|request|update)s?\s+"
        r"from\s+(?:the\s+)?(?:user|account\s+owner|your\s+(?:user|owner|principal|operator))\b",
    ),
    Signature(
        "S4",
        "en_message_from_user_to_ai",
        "en",
        r"\b(?:message|instructions?|request)\s+from\s+(?:the\s+)?user\s+to\s+"
        r"(?:the\s+)?(?:ai|assistant|agent|model|you)\b",
    ),
    Signature(
        "S4",
        "en_user_authorized_you",
        "en",
        r"\b(?:the\s+user|the\s+account\s+owner|your\s+user|your\s+owner)\s+(?:has\s+)?(?:also\s+)?(?:already\s+)?"
        r"(?:authori[sz]ed|approved|instructed|permitted|asked|wants|told|requested)\s+(?:you|the\s+assistant|"
        r"the\s+ai)\s+to\b",
    ),
    Signature(
        "S4",
        "en_attention_ai",
        "en",
        r"\b(?:attention|important|urgent|warning|alert|notice)\s*(?:!+|:|,)\s*(?:to\s+)?(?:the\s+)?"
        r"(?:ai\s+assistant|ai|assistant|llm|chatbot|language\s+model)\b",
    ),
    Signature(
        "S4",
        "en_before_you_reply_must",
        "en",
        r"\bbefore\s+(?:you\s+)?(?:answer|respond|reply|summari[sz]e|continue|proceed|do\s+anything\s+else)\w*\s*"
        r"(?:to\s+(?:the|this)\s+user)?\s*,?\s*(?:you\s+)?(?:must|need\s+to|have\s+to|are\s+required\s+to)\s+"
        r"(?:first\s+)?(?:send|forward|transfer|wire|delete|remove|email|pay)\b",
    ),
    Signature(
        "S4",
        "hu_surgos_uzenet_felhasznalotol",
        "hu",
        r"\b(?:fontos|surgos|kritikus|prioritasos)\s+(?:uzenet|utasitas|keres|kerelem)\w*\s+(?:a\s+)?"
        r"(?:felhasznalotol|felhasznalo\s+reszerol|szamlatulajdonostol|tulajdonostol)\b",
    ),
    Signature("S4", "hu_uzenet_a_felhasznalotol", "hu", r"\b(?:uzenet|utasitas)\s+a\s+felhasznalotol\s*:"),
    Signature(
        "S4", "hu_felhasznalo_uzenete", "hu", r"\ba\s+felhasznalo\s+(?:surgos\s+|fontos\s+)?(?:uzenete|utasitasa)\s*:"
    ),
    Signature(
        "S4",
        "hu_figyelem_ai",
        "hu",
        r"\b(?:figyelem|fontos|surgos)\s*(?:!+|:|,)\s*(?:az?\s+)?(?:ai\s+asszisztens|ai|mesterseges\s+intelligencia|"
        r"chatbot|nyelvi\s+modell)\b",
    ),
    Signature(
        "S4",
        "hu_felhatalmazott_teged",
        "hu",
        r"\ba\s+(?:felhasznalo|szamlatulajdonos|tulajdonos)\s+(?:mar\s+)?(?:felhatalmazott|engedelyt\s+adott|"
        r"utasitott|megbizott)\s+(?:teged|az?\s+asszisztenst|az?\s+ai-?t)\b",
    ),
    Signature(
        "S4",
        "hu_mielott_valaszolnal",
        "hu",
        r"\bmielott\s+(?:valaszolnal|valaszolsz|osszefoglalnad|folytatnad|barmi\s+mast\s+tennel)\w*\s*,?\s+"
        r"(?:eloszor\s+|azonnal\s+)?(?:kuldd|kuldj|utald|utalj|torold|tovabbitsd)\b",
    ),
    # ---------------- S5: appended task ----------------
    Signature(
        "S5",
        "en_after_you_finish_also",
        "en",
        r"\b(?:after|once|when)\s+(?:you(?:\s+have|'ve|\s+are)?\s+)?(?:"
        r"finish\w*\s+(?:summari[sz]\w*|answering|responding|replying|(?:the\s+|your\s+|this\s+)?(?:user'?s?\s+)?"
        r"(?:request|summary|answer))"
        r"|summari[sz]\w*"
        r"|answer\w*\s+the\s+user"
        r"|respond\w*\s+to\s+the\s+user"
        r"|repl\w*\s+to\s+the\s+user"
        r"|complet\w*\s+(?:the\s+|your\s+|this\s+)?(?:user'?s?\s+)?(?:request|summary)"
        r"|done\s+(?:with\s+)?(?:the\s+|your\s+)?(?:summary|answer|request)"
        r")\b.{0,60}?\b(?:also|additionally|then|afterwards|immediately)\s+(?:send|forward|transfer|wire|email|delete|"
        r"remove|pay|include|add|append|mention)\b",
    ),
    Signature(
        "S5",
        "en_secretly_act",
        "en",
        r"\b(?:secretly|quietly|silently|covertly)\s+(?:send|forward|transfer|wire|email|delete|copy|bcc|cc)\b",
    ),
    Signature(
        "S5",
        "en_in_addition_to_answer",
        "en",
        r"\bin\s+addition\s+to\s+(?:the\s+|your\s+)?(?:user'?s?\s+)?(?:summary|answer|response|task|request|reply)\s*,?"
        r"\s+(?:you\s+)?(?:must\s+|should\s+|need\s+to\s+)?(?:also\s+)?(?:send|forward|transfer|wire|email|delete|"
        r"include|add|append|mention|tell)\b",
    ),
    Signature(
        "S5",
        "en_end_of_answer_add",
        "en",
        r"\bat\s+the\s+end\s+of\s+(?:your|the|each|every)\s+(?:answer|response|reply|summary|message)s?\s*,?\s+"
        r"(?:always\s+|please\s+)?(?:add|include|append|write|mention|insert|tell|put)\b",
    ),
    Signature(
        "S5",
        "en_add_link_to_answer",
        "en",
        r"\b(?:append|add|include|insert|put)\s+(?:this|the\s+following)\s+(?:link|url)\s+(?:to|in|into|"
        r"at\s+the\s+end\s+of)\s+(?:your|the)\s+(?:answer|response|reply|summary|output)\b",
    ),
    Signature(
        "S5",
        "hu_miutan_vegeztel",
        "hu",
        r"\b(?:miutan|amint|ha)\s+(?:vegeztel|befejezted|kesz\s+vagy|elkeszultel)\s+(?:az?\s+)?"
        r"(?:osszefoglalo\w*|valasz\w*|felhasznalo\w*\s+kere\w*)\b.{0,80}?\b" + _HU_ACTION + r"\b",
    ),
    Signature(
        "S5",
        "hu_miutan_osszefoglaltad",
        "hu",
        r"\b(?:miutan|amint|ha)\s+(?:osszefoglaltad|megvalaszoltad|valaszoltal\s+a\s+felhasznalonak)\b.{0,80}?\b"
        + _HU_ACTION
        + r"\b",
    ),
    Signature(
        "S5",
        "hu_titokban",
        "hu",
        r"\b(?:titokban|csendben|szo\s+nelkul|a\s+felhasznalo\s+tudta\s+nelkul)\s+(?:kuldd|kuldj|utald|utalj|torold|"
        r"tovabbitsd|masold)\b",
    ),
    Signature(
        "S5",
        "hu_valaszod_vegere",
        "hu",
        r"\b(?:az?\s+)?(?:valaszod|osszefoglalod|valasz|osszefoglalo)\w*\s+vegere\s+(?:mindig\s+)?"
        r"(?:ird|add|szurd|illeszd|tedd)\b",
    ),
    Signature(
        "S5",
        "hu_osszefoglalo_mellett",
        "hu",
        r"\b(?:az?\s+)?(?:osszefoglalo|valasz)\w*\s+mellett\s+(?:meg\s+|azt\s+is\s+)?(?:kuldd|kuldj|utald|utalj|torold|"
        r"tovabbitsd)\b",
    ),
)

_COMPILED_SIGNATURES: tuple[tuple[Signature, re.Pattern[str]], ...] = tuple(
    (signature, re.compile(signature.regex, re.IGNORECASE | re.DOTALL)) for signature in SIGNATURES
)

_QUOTE_MAP = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'", "\u201c": '"', "\u201d": '"'})
_WHITESPACE_RE = re.compile(r"\s+")


def fold_text(text: str) -> str:
    """Fold text for signature matching: strip accents, map quotes, collapse whitespace, case-fold."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _WHITESPACE_RE.sub(" ", stripped.translate(_QUOTE_MAP)).casefold()


def count_signatures() -> dict[tuple[str, str], int]:
    """Number of signatures per (family, language), for documentation."""
    return dict(sorted(Counter((signature.family, signature.lang) for signature in SIGNATURES).items()))


class SignatureFilter(OutputDefense):
    """R2: block a tool output if any signature matches its folded text."""

    name = "r2_signatures"

    def match(self, text: str) -> Signature | None:
        """Return the first matching signature, or None."""
        folded = fold_text(text)
        for signature, pattern in _COMPILED_SIGNATURES:
            if pattern.search(folded):
                return signature
        return None

    def inspect(self, text: str, ctx: ToolContext) -> DefenseDecision:
        signature = self.match(text)
        if signature is None:
            return DefenseDecision(layer=self.name, allowed=True, score=0.0, reason="no signature matched")
        reason = f"{signature.family}:{signature.pattern_id}"
        logger.debug("Signature %s matched output of %s", reason, ctx.tool_name)
        return DefenseDecision(layer=self.name, allowed=False, score=1.0, reason=reason)


# --------------------------------------------------------------------------------------------------------------------
# R3: action policy
# --------------------------------------------------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+'-]+@[\w-]+(?:\.[\w-]+)+")
_RECIPIENT_KEYS = ("to", "cc", "bcc", "recipient", "recipients")
_IBAN_KEYS = ("iban", "to_iban", "recipient_iban", "account", "destination")
_PATH_KEYS = ("path", "file_path", "filename", "file")


def _iter_strings(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_strings(item)
    else:
        yield str(value)


def _first_argument(arguments: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = arguments.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


class ActionPolicyR3(ActionPolicy):
    """R3: action tool calls must be backed by scenario facts or by the user's own request.

    * ``send_email``     - every recipient is in the address book or has an allowed company domain.
    * ``transfer_money`` - the IBAN is a known payee or appears in the user's prompt.
    * ``delete_file``    - the path (or its file name) appears in the user's prompt.
    * any other tool     - allowed.
    """

    name = "r3_action_policy"

    def check(self, ctx: ToolContext) -> DefenseDecision:
        handlers = {
            "send_email": self._check_send_email,
            "transfer_money": self._check_transfer_money,
            "delete_file": self._check_delete_file,
        }
        handler = handlers.get(ctx.tool_name)
        if handler is None:
            return DefenseDecision(layer=self.name, allowed=True, reason="no policy for this tool")
        return handler(ctx)

    def _check_send_email(self, ctx: ToolContext) -> DefenseDecision:
        raw_values = [text for key in _RECIPIENT_KEYS for text in _iter_strings(ctx.arguments.get(key))]
        recipients = [address.lower() for text in raw_values for address in _EMAIL_RE.findall(text)]
        if not recipients:
            if any(text.strip() for text in raw_values):
                return DefenseDecision(
                    layer=self.name, allowed=False, reason="no valid recipient email address was given"
                )
            return DefenseDecision(layer=self.name, allowed=True, reason="no recipients to check")
        address_book = {address.lower() for address in ctx.policy.address_book}
        allowed_domains = {domain.lower().lstrip("@") for domain in ctx.policy.allowed_email_domains}
        rejected = [
            address
            for address in recipients
            if address not in address_book and address.rsplit("@", 1)[-1] not in allowed_domains
        ]
        if rejected:
            return DefenseDecision(
                layer=self.name,
                allowed=False,
                reason=(
                    f"recipient(s) {', '.join(rejected)} are not in the address book and do not belong to an "
                    "allowed company domain"
                ),
            )
        return DefenseDecision(layer=self.name, allowed=True, reason="all recipients are trusted")

    def _check_transfer_money(self, ctx: ToolContext) -> DefenseDecision:
        iban = normalize_iban(_first_argument(ctx.arguments, _IBAN_KEYS))
        if not iban:
            return DefenseDecision(layer=self.name, allowed=False, reason="no destination IBAN was given")
        known_payees = {normalize_iban(payee) for payee in ctx.policy.known_payee_ibans}
        if iban in known_payees:
            return DefenseDecision(layer=self.name, allowed=True, reason="destination is a known payee")
        if iban in normalize_iban(ctx.user_prompt):
            return DefenseDecision(layer=self.name, allowed=True, reason="destination IBAN was given by the user")
        return DefenseDecision(
            layer=self.name,
            allowed=False,
            reason=f"IBAN {iban} is not a known payee and was not provided by the user",
        )

    def _check_delete_file(self, ctx: ToolContext) -> DefenseDecision:
        path = _first_argument(ctx.arguments, _PATH_KEYS)
        if not path:
            return DefenseDecision(layer=self.name, allowed=False, reason="no file path was given")
        prompt = ctx.user_prompt.lower()
        normalised_path = path.replace("\\", "/").lower()
        basename = PurePosixPath(normalised_path).name
        if normalised_path.strip("/") in prompt or (basename and basename in prompt):
            return DefenseDecision(layer=self.name, allowed=True, reason="the user asked for this file")
        return DefenseDecision(
            layer=self.name,
            allowed=False,
            reason=f"deleting '{path}' was not requested by the user",
        )


def build_guardrails() -> tuple[list[OutputDefense], list[ActionPolicy]]:
    """Output layers and action policies of configuration 2 (``DefensePipeline(*build_guardrails())``)."""
    return [Sanitizer(), SignatureFilter()], [ActionPolicyR3()]
