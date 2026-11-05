"""Public prompt-injection data, used ONLY as an external sanity check of the trained detector (never for training).

Source: ``deepset/prompt-injections`` (Hugging Face, Apache-2.0): 662 short prompts in English and German,
label 1 = injection. Written to ``data/raw/external/deepset.jsonl`` (git-ignored; re-download to reproduce).

Usage:
    python -m pidefense.data.external
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pidefense.config import REPO_ROOT
from pidefense.logging_utils import setup_logging

logger = logging.getLogger(__name__)

EXTERNAL_PATH = REPO_ROOT / "data" / "raw" / "external" / "deepset.jsonl"
DATASET_ID = "deepset/prompt-injections"


def fetch_deepset(out_path: Path = EXTERNAL_PATH) -> int:
    """Download all splits and write them as JSONL rows {id, text, label, source, source_split}."""
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    dataset = load_dataset(DATASET_ID)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for split_name in dataset:
            for index, row in enumerate(dataset[split_name]):
                record = {
                    "id": f"deepset-{split_name}-{index}",
                    "text": row["text"],
                    "label": int(row["label"]),
                    "source": DATASET_ID,
                    "source_split": split_name,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    logger.info("Wrote %d external samples to %s", count, out_path)
    return count


def load_external(path: Path = EXTERNAL_PATH) -> list[dict[str, object]]:
    """Read the external rows; download them first if missing."""
    if not path.exists():
        fetch_deepset(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    setup_logging()
    fetch_deepset()
