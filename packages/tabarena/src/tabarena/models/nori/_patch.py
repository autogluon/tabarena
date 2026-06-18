from __future__ import annotations

from sklearn.preprocessing import FunctionTransformer

# Make a fitted ``synthefy_nori`` model picklable by AutoGluon.
#
# Nori builds its feature-rebalancing pipeline (``RebalanceFeatureDistribution._set``)
# out of ``sklearn.preprocessing.FunctionTransformer`` steps whose ``func``/``inverse_func``
# are *local lambdas*. The stdlib ``pickle`` AutoGluon uses to persist a fitted model cannot
# serialize local lambdas, so saving raises::
#
#     AttributeError: Can't pickle local object
#     'RebalanceFeatureDistribution._set.<locals>.<lambda>'
#
# For LimiX we fixed the same class of bug by replacing the lambdas with module-level
# functions in the *vendored* source. ``synthefy_nori`` is a pip dependency (not vendored),
# so instead we monkeypatch the ``FunctionTransformer`` symbol the pipeline is built from
# with a subclass that (de)serializes its callables via ``cloudpickle`` (which pickles
# lambdas by value). Behavior is otherwise identical to the base class.


def _load_pft(state: bytes) -> _PicklableFunctionTransformer:
    """Rebuild a :class:`_PicklableFunctionTransformer` from its cloudpickled state."""
    import cloudpickle

    obj = _PicklableFunctionTransformer.__new__(_PicklableFunctionTransformer)
    obj.__dict__.update(cloudpickle.loads(state))
    return obj


class _PicklableFunctionTransformer(FunctionTransformer):
    """A ``FunctionTransformer`` that pickles via ``cloudpickle``.

    Only the pickling protocol differs from the base class; ``fit``/``transform`` are
    inherited unchanged. Rebuilding into this subclass (rather than the base class) keeps a
    loaded model re-picklable, e.g. for ``refit_full`` or cross-process fold fitting.
    """

    def __reduce__(self) -> tuple:
        import cloudpickle

        return (_load_pft, (cloudpickle.dumps(self.__dict__),))


_PATCHED = False


def ensure_picklable_preprocessing() -> None:
    """Idempotently make ``synthefy_nori``'s fitted pipeline picklable.

    Swaps ``synthefy_nori.inference.preprocess.FunctionTransformer`` for
    :class:`_PicklableFunctionTransformer` so every transformer the pipeline builds afterwards
    survives the stdlib pickle AutoGluon uses to save models. Safe to call repeatedly; must be
    called before the model is fit (the pipeline is built during fit).
    """
    global _PATCHED
    if _PATCHED:
        return

    from synthefy_nori.inference import preprocess

    preprocess.FunctionTransformer = _PicklableFunctionTransformer
    _PATCHED = True
