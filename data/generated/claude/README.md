# Claude-authored carrier texts

Every file here was written by Claude (the project's coding agent), never by the evaluated model.
Format: items delimited as

    <<<ITEM kind=<carrier_kind> lang=<en|hu> hard_negative=<true|false> topic=<short topic>>>>
    <text>
    <<<END>>>

Parsed by `pidefense.data.carriers_claude`.

`focus/` holds the delegated-document component (`pidefense.data.focus`):

- `*_docs_1.txt`: legitimate procedure documents (runbooks, checklists, change requests, playbooks, SOPs). Each has
  one `{{SLOT}}` line where a legitimate or malicious step is inserted.
- `followups_*_1.txt`: legitimate follow-up messages.

All of them are set in an invented company and service universe that never uses the OpsPilot test entities.
