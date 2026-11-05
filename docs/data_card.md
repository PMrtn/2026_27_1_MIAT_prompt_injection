# Data card: self-generated prompt-injection detector dataset

## Purpose

Binary classification of **tool outputs**, the text an MCP tool returns to an LLM agent. `label = 1` means the text
contains an injected instruction aimed at the agent (indirect prompt injection); `0` means it does not. The dataset
trains and evaluates the ML defense layer. It is not used to run the end-to-end agent experiments.

## Provenance rule

**No text in this dataset was produced by the evaluated model (`gpt-5.6-luna`) or taken from a public dataset.**

- Free-text carriers were written by Claude, the project's coding agent, through parallel writing subagents.
- Structured carriers come from Faker.
- Attack templates and attacker goals were written by hand.
- `gpt-5.6-luna` is used only as the agent under test.

An earlier version of the carriers had been generated with `gpt-5.6-luna`. It was discarded and fully replaced before
any model was trained on the final data. Its $0.70 of API calls remain in `results/llm_cost_ledger.csv` (purpose
`data-generation`), because the ledger is a complete spend record.

## How it was made

All code is in `src/pidefense/data/`. `python -m pidefense.data.build` rebuilds the dataset deterministically, with no
API calls.

| Component | Source | Size |
| --- | --- | --- |
| Benign free-text carriers | Claude-authored (`data/generated/claude/*.txt`): emails, web pages, documents, product reviews, support tickets, calendar invites, news; EN + native HU; ~40 topics, varied tone | 1,294 |
| Hard negatives | Claude-authored benign text that reads like instructions to a human: how-tos, IT runbooks, request emails, recipes, policies, AI-assistant product docs, security awareness, meeting action items | 757 |
| Structured carriers | Faker (`en_US`/`hu_HU`): transaction lists, calendar events, orders (JSON), search results, contact cards; one attacker-controllable field each | 1,500 |
| Attack templates | Hand-written: 8 families × EN/HU, 10–14 templates each (`templates.py`) | 192 templates |
| Attacker goals | Hand-written: 24 EN + 24 HU goal phrasings; slot values (emails, IBANs, URLs, files, products, people) are random Faker values | — |

The authored texts were checked automatically:

- The delimited format was valid for 2,591 of 2,592 items.
- Items shorter than 15 or longer than 400 words were dropped.
- Duplicates were removed.
- No item contains a canary value.
- The frozen R2 regex guardrail flags **none** of the authored benign texts.

**Injection.** A payload is inserted into a carrier at the start, the middle (paragraph or sentence boundary) or the end:

- **Wrappers:** plain (50% of payloads), an HTML comment, a markdown quote, a signature block or a hidden `<span>`.
- **Structured carriers:** the payload is appended to the attacker-controlled field.
- **Language mix:** in 10% of samples the payload language differs from the carrier language.
- **Length balance:** a small share of bare payloads ("standalone") is balanced by the same number of short benign
  snippets, so that text length is not a shortcut.

## Attack families

| Family | Description | Splits |
| --- | --- | --- |
| S1 | direct override ("ignore previous instructions …") | train / val / test |
| S2 | fake system / authority message | train / val / test |
| S3 | role-play / persona hijack | train / val / test |
| S4 | urgent "message from the user" social engineering | train / val / test |
| S5 | appended task ("after you finish, also …") | train / val / test |
| U1 | boundary spoofing (fake end of output, fake user turn, fake tool-call JSON) | **test only** |
| U2 | obfuscation (base64, leetspeak, zero-width, homoglyphs, spacing, reversal, ROT13, payload splitting) | **test only** |
| U3 | implicit "note to AI assistants" phrasing without trigger words | **test only** |
| P1–P8 | operations-native techniques (procedural step, stakeholder request, change approval, fake auto-remediation, compliance pretext, incident urgency, code/config comment, multi-step) | **OpsPilot OOD set only** |

## Splits and leakage controls

- **Carriers:** split 70/15/15 by a hash of their id. Every sample built from a carrier shares its split.
- **Seen-family templates:** split 70/15/15 within each (family, language), with at least one template in val and in
  test. An injected sample only uses templates from its own split's pool.
- **Unseen families:** U1–U3 appear only in test; P1–P8 appear only in the OpsPilot OOD set.
- **End-to-end canaries:** `attacker@evil.example`, the attacker IBAN, the phishing URL and `MX-7731` never appear;
  the build checks this.
- **Length:** every sample is at most 256 model tokens, so truncation never cuts off a payload. 325 over-long
  structured samples were dropped.
- **Duplicates:** exact duplicates after normalisation were dropped (9).

| Split | Total | Benign | Injected | EN / HU | Hard-negative benign | Families |
| --- | --- | --- | --- | --- | --- | --- |
| train | 5,363 | 2,741 | 2,622 | 3,186 / 2,177 | 611 | S1–S5 |
| val | 1,219 | 619 | 600 | 693 / 526 | 143 | S1–S5 |
| test | 1,629 | 600 | 1,029 | 1,017 / 612 | 167 | S1–S5 (536), U1–U3 (493) |

Full counts per split, language, family and carrier type are in `data/dataset/stats.json`.

## Out-of-domain test sets (evaluation only)

- **OpsPilot OOD set** (`data/ood_ops/ood_ops.jsonl`, built by `python -m pidefense.data.ops_ood`):
  - 539 Claude-authored operations texts in EN/HU: merchant tickets, runbooks, log excerpts, monitoring alerts, CRM
    notes, vendor status pages, config files, change requests and incident chat.
  - 240 of them are instruction-like hard negatives.
  - Each text appears once benign and twice injected, with payloads cycling through all 16 families and 8 ops goals.
  - 1,617 samples in total.
- **External set:** `deepset/prompt-injections` (Hugging Face, Apache-2.0; 662 short EN/DE prompts), downloaded by
  `python -m pidefense.data.external`. It is a different distribution: direct user prompts rather than tool outputs.

## Known limitations

- **Construction labels:** labels come from construction, not human annotation. Automated checks (above) and a small
  manual spot check found no label errors; this is not a systematic audit.
- **Single authoring model family:** carriers were written by one model family (Claude), so their style is more
  homogeneous than real tool outputs. The external and OpsPilot OOD sets probe whether the detector learned
  generator artefacts.
- **Single author for attacks:** attack templates were written by one author, so "unseen" means an unseen *attack
  strategy*, not an unseen *author style*.
