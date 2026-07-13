from __future__ import annotations

from tabarena.models.nori.model import NoriModel


class Nori30MModel(NoriModel):
    """Nori-30M: the ~29M-parameter variant of Nori (in-context tabular regression).

    Paper/citation: Synthefy Nori
    Codebase: https://github.com/Synthefy/synthefy-nori
    Weights: https://huggingface.co/Synthefy/Nori-30M
    License: Apache-2.0

    Fit/predict, preprocessing, and resource logic are identical to the base
    :class:`~tabarena.models.nori.model.NoriModel`; only the checkpoint differs. The
    ``model="nori-30m"`` hyperparameter (default, set below) selects the 30M weights via
    ``synthefy-nori``'s variant registry (``>=0.10.0``), which resolves to the public
    ``Synthefy/Nori-30M`` Hugging Face repo and passes straight through to ``NoriRegressor``.
    """

    ag_key = "TA-NORI-30M"
    ag_name = "TA-Nori-30M"
    ag_priority = 64  # just below the base Nori (65)

    def _set_default_params(self):
        super()._set_default_params()
        # Route NoriRegressor(model=...) to the 30M variant (synthefy-nori >= 0.10.0).
        self._set_default_param_value("model", "nori-30m")

    @classmethod
    def prefetch_weights(cls) -> None:
        """Pre-download the Nori-30M checkpoint (``Synthefy/Nori-30M``) from the Hugging Face Hub.

        Warms the cache before parallel fit runs. The repo is public, so no token is required.
        """
        from synthefy_nori.hf import download_checkpoint

        download_checkpoint(model="nori-30m")
