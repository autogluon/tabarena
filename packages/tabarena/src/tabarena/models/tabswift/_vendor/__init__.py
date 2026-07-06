"""Vendored TabSwift source.

Upstream: https://github.com/LAMDA-Tabular/TabSwift (MIT), the ``TALENT/model/lib/tabswift``
package. TabSwift is not published as a pip-installable package, so the inference-path
sources are vendored here verbatim (apart from the small, documented tweaks in
``README.md``). All upstream imports are relative (``from .model.tabswift import ...``),
so they resolve unchanged under the ``tabarena.models.tabswift._vendor`` namespace.

The public symbols are imported lazily by ``tabarena.models.tabswift.model`` inside
``_fit`` (``from tabarena.models.tabswift._vendor.classifier import TabSwiftClassifier``),
so importing this package never eagerly pulls in torch.
"""

from __future__ import annotations
