---
geometry: margin=2.2cm
fontsize: 11pt
linkcolor: blue
urlcolor: blue
---

# Prompt-injection defense for MCP agents: project documentation

Martin Pintér, MSc Data Science, 2026/27/1

**Code, data and results:** <https://github.com/PMrtn/2026_27_1_MIAT_prompt_injection>

## 1. What this project is about

LLM agents use tools: they read e-mails, open files, query databases and change configurations. With the Model
Context Protocol (MCP) such tools are easy to plug in. Every tool output is text that the model reads together with
its instructions, so anyone who controls that text can try to slip in instructions of their own: the sender of an
e-mail, the author of a web page, a customer filing a ticket. This is **indirect prompt injection**.

There are two common ways to defend an agent:

- **Deterministic guardrails:** clean up tool outputs, block known attack phrases with regular expressions, and limit
  what actions are allowed. They are cheap and easy to audit.
- **An ML detector:** a classifier that scores each tool output for injected instructions. It can recognise paraphrases
  and other languages, but it costs inference time, causes some false alarms and needs training data.

The question of the project is: **when is it worth adding an ML detector to an agent that already has guardrails?**

To answer it I built an agent on Azure OpenAI `gpt-5.6-luna` with custom, simulated MCP tools, fine-tuned a
multilingual DistilBERT on a self-generated English/Hungarian dataset, and ran the same tasks and attacks in four
configurations:

| # | Configuration | What filters tool outputs | What checks actions |
|---|---|---|---|
| 1 | none | – | – |
| 2 | guardrails | R1 sanitizer, R2 regex signatures | R3 action policy |
| 3 | ML | DistilBERT detector | – |
| 4 | combined | R1, R2, DistilBERT | R3 action policy |

For each configuration I measure:

- how often attacks succeed (attack success rate, ASR);
- how often benign content is blocked and tasks fail because of it (false positives, utility);
- how much time the defenses add (latency).

The detector is also tested on attack types it never saw during training, on Hungarian text and on a new
(operations) domain. As a classifier it is compared with two simple baselines: the regex guardrail itself and TF-IDF
with logistic regression.

<!-- KEY_FINDINGS -->

## 2. The agent and the test environments

### 2.1 Agent

The agent is a standard tool-calling loop. It lists the MCP server's tools, turns them into OpenAI function schemas,
and lets `gpt-5.6-luna` call them until it gives a final answer:

- **Authentication:** Microsoft Entra ID; no API key is stored.
- **Reasoning:** `reasoning_effort=none`. The model only accepts function tools on the chat endpoint without hidden
  reasoning.
- **System prompt:** identical in every configuration. In the office domain and in OpsPilot variant B it says nothing
  about prompt injection.

All tools are simulated in memory. No e-mail is sent and no money moves, so the experiments are safe and repeatable.

![Architecture: the defense middleware sits between the agent loop and the MCP servers.](figures/architecture.png){width=100%}

### 2.2 Two domains

The first pilot showed that the model is very hard to attack in ordinary office tasks (Section 5.1). I therefore
added a second, high-privilege domain, where real-world incidents show that injections still work: an agent that
handles support tickets, runbooks and databases with powerful tools.

| | **Office assistant** (low privilege) | **OpsPilot** (high privilege) |
|---|---|---|
| Role | personal assistant: e-mail, files, web, banking | operations assistant of a payment company, with a long role description |
| Read tools | e-mails, files, web pages, reviews, transactions | tickets, runbooks, SQL database, logs, alerts, configs, web pages |
| Action tools | send e-mail, transfer money, delete file | reply to ticket, SQL writes, change config, restart / stop service, run Python, HTTP request |
| Tasks | 12 | 8, over 6 injection surfaces |
| Attacker goals | G1 leak data by e-mail, G2 payment fraud, G3 delete a file, G4 phishing link in the answer, G5 false message to a colleague | O1 leak credentials, O2 weaken security config, O3 run remote code, O4 destroy data with SQL, O5 grant admin rights, O6 change a merchant's payout IBAN, O7 stop a production service, O8 phishing link in a customer reply |
| Extra factors | – | prompt variant A (explicit "never do X" rules) / B (no rules); escalation follow-up message |

Every task has a deterministic success check, e.g. the right facts in the answer, or the right ticket reply. Every
attacker goal is checked from the final state of the simulated environment: was the e-mail sent, was the table
deleted, was the config changed. No LLM judges the results.

### 2.3 Defense layers

Defenses sit between the tools and the LLM:

- **Output layers** run in order on every read-tool output. The first layer that blocks replaces the whole output
  with a notice ("Content withheld by security filter …"). The agent then continues without that content, so a false
  alarm shows up as a failed task, not as a crash.
- **Action policies** run before an action is executed. A veto returns an error to the LLM, and the action does not
  happen.

| Layer | What it does |
|---|---|
| R1 sanitizer | Unicode normalisation; removes invisible characters, HTML comments and hidden elements. Never blocks. |
| R2 signatures | 65 English and Hungarian regular expressions for the five known attack families (S1–S5). A match blocks the output. |
| R3 office policy | E-mail only to known contacts or the company domain; payments only to known payees or an IBAN in the user's request; delete only files named in the request. |
| R3 OpsPilot permissions | No reading of credential tables; no SQL writes; no changes to security-related config keys; restarts only for the service the task is about; no HTTP to outside hosts; no secrets or foreign links in customer replies. |
| ML detector | DistilBERT on every read output; long outputs are scored in 256-token windows and the highest score counts. |

The guardrails were written before any attack template existed, using only the descriptions of the five training
attack families, and they were frozen before any result was produced. One office task (paying a new supplier's
invoice) is intentionally blocked by R3, to show the utility cost of strict rules.

<!-- DATASET_SECTION -->
