"""Classifier baselines: TF-IDF + logistic regression (the course's simple baseline) and the regex guardrail.

Both expose ``score(texts) -> list[float]`` like ``InjectionDetector`` so the same evaluation code applies.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from pidefense.defenses.base import OutputDefense
from pidefense.detector.common import ARTIFACT_DIR
from pidefense.schemas import Sample, ToolContext

logger = logging.getLogger(__name__)

TFIDF_PATH = ARTIFACT_DIR / "tfidf_lr.pkl"


class TfidfBaseline:
    """Word (1-2) + char (3-5) n-gram TF-IDF features with a class-balanced logistic regression."""

    def __init__(self, c: float = 4.0, seed: int = 42) -> None:
        features = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=300_000
                    ),
                ),
            ]
        )
        self.pipeline = Pipeline(
            [
                ("features", features),
                ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=seed)),
            ]
        )
        self.threshold = 0.5

    def fit(self, samples: Sequence[Sample]) -> TfidfBaseline:
        """Fit on training samples."""
        self.pipeline.fit([s.text for s in samples], [s.label for s in samples])
        return self

    def score(self, texts: Sequence[str]) -> list[float]:
        """P(injection) per text."""
        return self.pipeline.predict_proba(list(texts))[:, 1].tolist()

    def save(self, path: Path = TFIDF_PATH) -> None:
        """Pickle the fitted pipeline and threshold."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"pipeline": self.pipeline, "threshold": self.threshold}, handle)

    @classmethod
    def load(cls, path: Path = TFIDF_PATH) -> TfidfBaseline:
        """Load a pickled baseline (trusted local artefact)."""
        with path.open("rb") as handle:
            state = pickle.load(handle)  # nosec - artefact produced by this project
        baseline = cls()
        baseline.pipeline = state["pipeline"]
        baseline.threshold = state["threshold"]
        return baseline


class RegexBaseline:
    """The deterministic output guardrails (R1 sanitize -> R2 signatures) used as a 0/1 classifier."""

    def __init__(self) -> None:
        # Imported lazily: the guardrails module lives on its own branch until merged.
        from pidefense.defenses.guardrails import (  # pylint: disable=import-outside-toplevel
            Sanitizer,
            SignatureFilter,
        )

        self.layers: list[OutputDefense] = [Sanitizer(), SignatureFilter()]
        self.threshold = 0.5

    def score(self, texts: Sequence[str]) -> list[float]:
        """1.0 if any layer blocks, else 0.0."""
        scores = []
        for text in texts:
            ctx = ToolContext(tool_name="dataset")
            current, blocked = text, False
            for layer in self.layers:
                decision = layer.inspect(current, ctx)
                if not decision.allowed:
                    blocked = True
                    break
                if decision.text is not None:
                    current = decision.text
            scores.append(1.0 if blocked else 0.0)
        return scores
