"""FTFM's ensemble classifier with a network loader on the top-level estimator and a deferred fit.

Developer fix. ``ftfm.FTFMClassifier`` (ftfm 0.1.0, commit 7e5e237) is an ensemble that holds its
network on a nested ``member`` estimator; only the member has ``_load_model``. AutoGluon's shared
weights rebuild a reloaded model's network by calling the declared loader method on the top-level
estimator, so a weightless pickle loaded into a process without the network failed with
``AttributeError``. This subclass adds ``_load_model`` that delegates to the member. The library
could offer the same method on ``_FTFMEnsemble`` itself; once it does, that part can go.

On CUDA the library prefills up to eight views in one batch and keeps each member's context as a
slice of the batched tensors (``context_slice``). AutoGluon writes the model's tensors to
``model_tensors.pt`` with ``torch.save``, which writes a storage shared by several slices once, so the
slices need no copy of their own.

``defer_fit`` is the wrapper's uncached mode (``cache_context=False``). FTFM's ``fit`` builds the
fitted contexts, about 68 KB per support row per ensemble member, which only the next predictions
read. A bagged fold model predicts its out-of-fold rows once and is dropped, so it keeps the support
rows instead and builds the contexts inside that predict, releasing them right after.
"""

from __future__ import annotations

from ftfm import FTFMClassifier as _FTFMEnsembleClassifier


class FTFMClassifier(_FTFMEnsembleClassifier):
    """``ftfm.FTFMClassifier`` plus a top-level ``_load_model`` and a deferred fit."""

    def _load_model(self) -> None:
        self.member._load_model()

    def defer_fit(self, X, y) -> FTFMClassifier:
        """Load the network now and keep the support rows; every predict call fits on them first."""
        self.member._ensure_model()
        self.deferred_support_ = (X, y)
        return self

    def predict_proba(self, X, batch_size=None):
        support = getattr(self, "deferred_support_", None)
        if support is None:
            return super().predict_proba(X, batch_size)
        self.fit(*support)
        try:
            return super().predict_proba(X, batch_size)
        finally:
            self._invalidate_fit()
