from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

#: Values of the ``reg_target_transform`` hyperparameter.
TARGET_TRANSFORMS = ("none", "log1p", "quantile")
#: Smallest and largest exponent the log1p smearing factor and the inverse are allowed to use.
_EXP_CLIP = 700.0
_RESIDUAL_CLIP = 20.0


class TargetTransform:
    """A monotone, invertible map of the regression target, fitted on training targets only.

    ``log1p`` maps ``y`` to ``log(1 + y)`` and back through ``expm1``, with a multiplicative
    smearing factor ``S`` (Duan, 1983) so that the back-transform estimates the conditional mean
    rather than the conditional median: ``E[y | x] = exp(f(x)) * S - 1`` with
    ``S = mean(exp(residual))``. The test metric is RMSE on the original scale, which rewards the
    mean. ``quantile`` maps ``y`` to a standard normal score through its fitted empirical
    quantiles and back through the same quantiles, so predictions stay inside the training range.
    It has no smearing: its inverse returns a value nearer the conditional median than the mean
    when the conditional distribution is skewed.

    Attributes:
        name: ``"log1p"`` or ``"quantile"``.
    """

    def __init__(self, name: str, quantiles=None):
        self.name = name
        self._quantiles = quantiles

    @classmethod
    def fit(cls, name: str, y: np.ndarray, seed: int = 0) -> tuple[TargetTransform | None, str | None]:
        """Fit the transform on training targets.

        Args:
            name: One of :data:`TARGET_TRANSFORMS`.
            y: Training targets.
            seed: Seed of the quantile transformer's subsample.

        Returns:
            ``(transform, None)``, or ``(None, reason)`` when ``name`` is ``"none"`` (reason
            ``None``) or the target does not admit the transform: ``log1p`` with a value below 0,
            or ``quantile`` with a constant or non-finite target.
        """
        import numpy as np

        if name == "none":
            return None, None
        y = np.asarray(y, dtype=np.float64)
        if not np.isfinite(y).all():
            return None, f"{name} refused: the target has non-finite values"
        if name == "log1p":
            if (y < 0).any():
                return (
                    None,
                    f"log1p refused: the target has {int((y < 0).sum())} value(s) below 0 (minimum {y.min():.6g})",
                )
            return cls(name), None
        if name == "quantile":
            if np.unique(y).size < 2:
                return None, "quantile refused: the target is constant"
            from sklearn.preprocessing import QuantileTransformer

            qt = QuantileTransformer(
                n_quantiles=min(1000, y.size),
                output_distribution="normal",
                subsample=max(10_000, y.size),
                random_state=int(seed),
            ).fit(y.reshape(-1, 1))
            return cls(name, quantiles=qt), None
        raise ValueError(f"reg_target_transform must be one of {TARGET_TRANSFORMS}, got {name!r}")

    def forward(self, y: np.ndarray) -> np.ndarray:
        """The target in the space the readout is trained in."""
        import numpy as np

        y = np.asarray(y, dtype=np.float64)
        if self.name == "log1p":
            return np.log1p(y)
        return self._quantiles.transform(y.reshape(-1, 1)).ravel()

    def inverse(self, z: np.ndarray, smear: float = 1.0) -> np.ndarray:
        """Map predictions back to the original scale.

        Args:
            z: Predictions in the transformed space.
            smear: Duan's factor for ``log1p``; ignored by ``quantile``. ``1.0`` is the plain
                ``expm1``.

        Returns:
            Predictions on the original scale. ``log1p`` predictions are floored at 0, because
            a transform is only fitted on a non-negative target.
        """
        import numpy as np

        z = np.asarray(z, dtype=np.float64)
        if self.name == "log1p":
            out = np.expm1(np.minimum(z + math.log(smear), _EXP_CLIP))
            return np.maximum(out, 0.0)
        return self._quantiles.inverse_transform(z.reshape(-1, 1)).ravel()


def smearing_factor(residuals: np.ndarray) -> float:
    """Duan's smearing factor ``mean(exp(residual))`` of residuals in the log1p space.

    Args:
        residuals: Held-out residuals ``z_true - z_pred`` in the transformed space.

    Returns:
        The factor; ``1.0`` when there are no residuals or the mean is not finite.
    """
    import numpy as np

    r = np.asarray(residuals, dtype=np.float64)
    if r.size == 0:
        return 1.0
    s = float(np.mean(np.exp(np.clip(r, -_RESIDUAL_CLIP, _RESIDUAL_CLIP))))
    return s if math.isfinite(s) and s > 0 else 1.0
