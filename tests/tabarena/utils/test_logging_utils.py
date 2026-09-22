from __future__ import annotations

import logging

from tabarena.utils.logging_utils import root_handlers_preserved


def test_root_handlers_preserved_removes_added_handler():
    root = logging.getLogger()
    before = list(root.handlers)
    added = logging.StreamHandler()
    with root_handlers_preserved():
        root.addHandler(added)
        assert added in root.handlers
    assert root.handlers == before


def test_root_handlers_preserved_keeps_existing_handlers():
    root = logging.getLogger()
    existing = logging.NullHandler()
    root.addHandler(existing)
    try:
        with root_handlers_preserved():
            logging.info("module-level logging installs a root handler when none exists")
        assert existing in root.handlers
    finally:
        root.removeHandler(existing)
