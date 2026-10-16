"""Ordinal (1-5) classifiers that do not require a GPU.

The plan asks for an *ordinal* target rather than five unrelated classes.
Three interchangeable back-ends are provided so the geometry-only baseline and
the hybrid model share the same semantics:

* :class:`OrdinalCumulativeLogistic` -- K-1 binomial logistic models on the
  cumulative targets ``P(y > k)`` (the proportional-odds idea, no extra deps).
* :class:`OrdinalForest` -- a regularized tree ensemble plus ordinal
  post-processing (expected-value decoding from class probabilities).
* :class:`OrdinalLogitStatsmodels` -- a thin wrapper over
  ``statsmodels`` ``OrderedModel`` used when available (the plan's first
  preference); importing this module never requires statsmodels.

The torch CORAL head lives in :mod:`sutureqc.models` so this module stays
lightweight.
"""

from __future__ import annotations

import numpy as np

from . import config

try:  # optional
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover - environment dependent
    OrderedModel = None  # type: ignore[assignment]
    _HAS_STATSMODELS = False


def to_zero_based(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=int) - config.SCORE_MIN


def decode_expected(probabilities: np.ndarray) -> np.ndarray:
    """Expected value under the ordinal distribution, rounded and clipped."""
    scores = np.arange(config.SCORE_MIN, config.SCORE_MAX + 1, dtype=float)
    expected = probabilities @ scores
    return np.clip(np.rint(expected), config.SCORE_MIN, config.SCORE_MAX).astype(int)


def cumulative_targets(y: np.ndarray, n_scores: int = config.N_SCORES) -> np.ndarray:
    """Return an ``(m, K-1)`` matrix of ``1[y > k]`` cumulative indicators."""
    y = np.asarray(y, dtype=int)
    thresholds = np.arange(config.SCORE_MIN, config.SCORE_MIN + n_scores - 1)
    return (y[:, None] > thresholds[None, :]).astype(int)


class OrdinalCumulativeLogistic:
    """Proportional-odds-style ordinal logistic via K-1 binomial fits."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = config.SEED) -> None:
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._models: list = []
        self.classes_ = np.arange(config.SCORE_MIN, config.SCORE_MAX + 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalCumulativeLogistic":
        from sklearn.linear_model import LogisticRegression

        self._models = []
        z = cumulative_targets(y)
        for k in range(z.shape[1]):
            column = z[:, k]
            if column.min() == column.max():
                self._models.append(None)  # degenerate column -> constant
                continue
            model = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.seed)
            model.fit(X, column)
            self._models.append(model)
        return self

    def _cumulative(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        probs = np.zeros((m, len(self.classes_) - 1))
        for k, model in enumerate(self._models):
            if model is None:
                probs[:, k] = 0.0
            else:
                probs[:, k] = model.predict_proba(X)[:, 1]
        return probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        cum = self._cumulative(X)
        m, k = cum.shape
        probabilities = np.zeros((m, k + 1))
        probabilities[:, 0] = 1.0 - cum[:, 0]
        for i in range(1, k):
            probabilities[:, i] = cum[:, i - 1] - cum[:, i]
        probabilities[:, k] = cum[:, k - 1]
        probabilities = np.clip(probabilities, 1e-9, None)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return decode_expected(self.predict_proba(X))


class OrdinalForest:
    """Random-forest classifier with ordinal expected-value decoding."""

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int | None = None,
        class_weight: str | None = "balanced",
        seed: int = config.SEED,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.seed = seed
        self._model = None
        self.classes_ = np.arange(config.SCORE_MIN, config.SCORE_MAX + 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalForest":
        from sklearn.ensemble import RandomForestClassifier

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(X, np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("OrdinalForest must be fitted before predicting")
        raw = self._model.predict_proba(X)
        probabilities = np.zeros((raw.shape[0], config.N_SCORES))
        for column, cls in enumerate(self._model.classes_):
            probabilities[:, int(cls) - config.SCORE_MIN] = raw[:, column]
        probabilities = np.clip(probabilities, 1e-9, None)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return decode_expected(self.predict_proba(X))


class OrdinalLogitStatsmodels:
    """Ordinal logistic regression via statsmodels ``OrderedModel`` (optional)."""

    def __init__(self, distr: str = "logit") -> None:
        if not _HAS_STATSMODELS:
            raise NotImplementedError("statsmodels OrderedModel is not available")
        self.distr = distr
        self._result = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalLogitStatsmodels":
        assert OrderedModel is not None
        import pandas as pd

        frame = pd.DataFrame(np.asarray(X, dtype=float))
        self._result = OrderedModel(np.asarray(y, dtype=int), frame, distr=self.distr).fit(method="bfgs", disp=False)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("OrdinalLogitStatsmodels must be fitted before predicting")
        import pandas as pd

        frame = pd.DataFrame(np.asarray(X, dtype=float))
        return np.asarray(self._result.predict(frame), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return decode_expected(self.predict_proba(X))


def make_ordinal_model(name: str = "auto", seed: int = config.SEED):
    """Return an unfitted ordinal classifier by name.

    ``"auto"`` prefers statsmodels ordinal logistic, then cumulative logistic.
    ``"forest"`` returns the tree baseline required by the plan's fallback.
    """
    if name == "forest":
        return OrdinalForest(seed=seed)
    if name == "cumulative":
        return OrdinalCumulativeLogistic(seed=seed)
    if name == "statsmodels" and _HAS_STATSMODELS:
        return OrdinalLogitStatsmodels()
    return OrdinalCumulativeLogistic(seed=seed)
