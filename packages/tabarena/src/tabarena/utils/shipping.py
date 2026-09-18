"""Marks the moment an object is being pickled to ship to worker processes.

``parallel_for`` enters :func:`shipping_context` around ``ray.put`` so that pickling hooks can
send a leaner form than a disk pickle would need. Anything outside the context (a
``pickle.dump`` to disk, ``copy.deepcopy``) pickles the full object.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

_SHIPPING: ContextVar[bool] = ContextVar("tabarena_shipping", default=False)


def is_shipping() -> bool:
    """Whether the current pickling happens inside :func:`shipping_context`."""
    return _SHIPPING.get()


@contextlib.contextmanager
def shipping_context() -> Iterator[None]:
    """Mark pickling within the block as shipping to workers."""
    token = _SHIPPING.set(True)
    try:
        yield
    finally:
        _SHIPPING.reset(token)
