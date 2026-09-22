"""Keep a library call from changing the process's logging configuration."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def root_handlers_preserved() -> Iterator[None]:
    """Remove the root-logger handlers the block adds.

    A library that logs through the module-level ``logging.info`` / ``logging.warning`` functions
    (tabpfn-extensions at import, TabFM's checkpoint loader) installs a ``StreamHandler`` on a root
    logger that has none, which duplicates every later log line on stderr and fails the global-state
    check of AutoGluon's model tests. Wrap such imports and calls in this context to leave the root
    logger as it was.
    """
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)


def import_many_class_classifier() -> type:
    """Return ``tabpfn_extensions.many_class.ManyClassClassifier``, imported under :func:`root_handlers_preserved`."""
    with root_handlers_preserved():
        from tabpfn_extensions.many_class import ManyClassClassifier
    return ManyClassClassifier
