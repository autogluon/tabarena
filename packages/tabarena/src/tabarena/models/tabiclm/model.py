"""TabICL-M: TabICLv2 with source-aware handling of incomplete, multi-source tables.

TabICL-M continues the released TabICLv2 weights with a block-structured
missingness prior and three additions that read a row's missingness pattern as
its provenance: source-relative column values, observed-only row attention and a
pattern token. On complete data it equals TabICLv2; its gains are on tables merged
from several sources with different feature subsets and measurement offsets.

Paper / code: https://github.com/Sompote/TabICL-M (BSD-3-Clause, a fork of TabICL).
The ``tabicl-m`` distribution installs under the import name ``tabicl``, so this
wrapper reuses the TabICLv2 wrapper unchanged apart from the checkpoints, which are
downloaded once from the repository's public git-LFS storage.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

from tabarena.models.tabicl.model import TabICLv2Model

logger = logging.getLogger(__name__)

_LFS = "https://media.githubusercontent.com/media/Sompote/TabICL-M/65b617714f348acdd391485be65ccf6fb8b2fece/checkpoints/tabicl-m-sa-20k"
CHECKPOINTS = {
    "clf": ("tabicl-m-classifier-sa-20k.ckpt", f"{_LFS}/clf/step-10000.ckpt"),
    "reg": ("tabicl-m-regressor-sa-20k.ckpt", f"{_LFS}/reg/step-10000.ckpt"),
}


def _cache_dir() -> Path:
    return Path(os.environ.get("TABICLM_CACHE", Path.home() / ".cache" / "tabicl-m"))


def checkpoint_path(kind: str) -> str:
    """Local path of the TabICL-M checkpoint for ``kind`` in {"clf", "reg"}, downloading it once."""
    override = os.environ.get("TABICLM_CLF" if kind == "clf" else "TABICLM_REG")
    if override:
        return override
    name, url = CHECKPOINTS[kind]
    path = _cache_dir() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.log(20, f"Downloading TabICL-M checkpoint {name} from {url}")
        tmp = path.with_suffix(".part")
        if not url.startswith("https://"):
            raise ValueError(f"refusing to download TabICL-M weights from a non-https URL: {url}")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - scheme checked above
        tmp.rename(path)
    return str(path)


class TabICLMModel(TabICLv2Model):
    """TabICL-M evaluated under the TabICLv2 protocol (bagging, refit, GPU)."""

    ag_key = "TA-TABICLM"
    ag_name = "TA-TabICLM"
    _supported_problem_types = ["binary", "multiclass", "regression"]

    def _get_model_params(self) -> dict:
        params = super()._get_model_params()
        # ``model_path`` takes precedence over ``checkpoint_version`` in the tabicl estimators.
        params["model_path"] = checkpoint_path("reg" if self.problem_type == "regression" else "clf")
        return params

    @staticmethod
    def checkpoint_search_space() -> list[tuple[str, str]]:
        return [(CHECKPOINTS["clf"][0], CHECKPOINTS["reg"][0])]

    @classmethod
    def prefetch_weights(cls) -> None:
        for kind in ("clf", "reg"):
            checkpoint_path(kind)
