"""Vendored TabLDM source.

Upstream: the ``xiaomi-tabldm`` package (pip name ``tabldm``), Apache-2.0, Copyright Xiaomi
Corporation. Not published to PyPI (its own ``pyproject.toml``/``README`` only document a
local ``pip install .`` build), so the inference-path sources are vendored here verbatim
apart from the small, documented import-path tweaks below.

Layout mirrors upstream's ``tabldm/`` package with its top-level re-export layer flattened
away: ``_model/`` and ``_sklearn/`` sit directly under this package instead of inside a
nested ``tabldm/`` subpackage. All upstream intra-package code already uses relative imports
except for a handful of files in ``_sklearn/`` that self-referenced the top-level ``tabldm``
package (``from tabldm import InferenceConfig``, ``from tabldm._model.X import Y``); those
were rewritten to import directly from ``_model``'s defining submodules by absolute path
(``from tabarena.models.tabldm._vendor._model.inference_config import InferenceConfig``,
``from tabarena.models.tabldm._vendor._model.X import Y``, matching the ``limix`` vendor
tree's convention) rather than through a re-exporting ``__init__.py``, so this package
itself declares no imports and stays free of eager, heavy (torch) side effects.

The public estimators are imported lazily by ``tabarena.models.tabldm.model`` inside
``_fit`` (``from tabarena.models.tabldm._vendor._sklearn.classifier_enhanced import
TabLDMEnhancedClassifier``), so importing this package never eagerly pulls in torch.
"""

from __future__ import annotations
