# ML-Based Prompt Injection Defense for MCP Agents

MSc Data Science semester project (2026/27/1). The project measures the added value of a fine-tuned multilingual
DistilBERT prompt-injection detector next to deterministic guardrails, in an LLM agent that uses MCP tools.

Four configurations are compared on the same agent, tasks and attacks:

| # | Configuration | Defense layers |
|---|---------------|----------------|
| 1 | `none`        | none |
| 2 | `guardrails`  | R1 sanitizer, R2 regex signatures, R3 action policy |
| 3 | `ml`          | DistilBERT detector on tool outputs |
| 4 | `combined`    | guardrails + detector |

Metrics: attack success rate (ASR), false positive rate and benign utility, latency. A separate analysis covers attack
families held out of training and Hungarian attacks.

> Status: under active development. Results, the report and the slides will be linked here.

## Setup (WSL / Linux / macOS)

```sh
scripts/setup_env.sh            # creates ~/.venvs/miat-pi and installs pinned requirements
. ~/.venvs/miat-pi/bin/activate
pytest
```

The agent uses Azure OpenAI (`gpt-5.6-luna`) through Entra ID: run `az login` first; no API key is stored.
Every billed call is priced and appended to `results/llm_cost_ledger.csv`; calls are refused once the project
budget is reached. Smoke test (one tiny call):

```sh
python -m pidefense.llm "Reply with exactly: OK"
```

## Repository layout

```text
src/pidefense/
  config.py         Azure endpoint, pricing, budget
  schemas.py        shared data contracts (samples, defense decisions, episode logs)
  llm/              Azure client, cost ledger, exact-request cache
  defenses/         defense interfaces and the middleware pipeline
tests/              offline unit tests
scripts/            POSIX shell entry points
```
