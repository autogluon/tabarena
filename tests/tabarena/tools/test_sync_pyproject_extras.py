"""`superseded` model entries stay out of the installable pyproject extras.

A superseded entry pins a version the entry that replaced it excludes (TabDPT_GPU needs
`tabdpt<1.2`, TabDPT-Turbo needs `>=1.2.0`), so unioning the two into one extra would make it
unresolvable and break `pip install tabarena[...]`.
"""

from __future__ import annotations

import pytest

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.tools.sync_pyproject_extras import _expected_extras


class _DummyModel:
    # The tool derives the extra's name from the model class's module path, so spell out a
    # realistic one instead of inheriting this test module's.
    __module__ = "tabarena.models.dummy.model"


def _info(*, pip_extra: tuple[str, ...], superseded: bool = False) -> ModelInfo:
    return ModelInfo(
        model_cls=_DummyModel,
        search_space=lambda: None,
        method_metadata=MethodMetadata(method="Dummy"),
        pip_extra=pip_extra,
        superseded=superseded,
    )


@pytest.fixture
def registry(monkeypatch):
    """Point `_expected_extras` at a registry we control."""

    def _install(entries: dict[str, ModelInfo]) -> None:
        monkeypatch.setattr(
            "tabarena.tools.sync_pyproject_extras.get_model_registry",
            lambda: entries,
        )

    return _install


def test_superseded_pin_is_excluded(registry):
    """The current entry's pin survives; the superseded entry's conflicting pin does not."""
    registry(
        {
            "Current": _info(pip_extra=("dummy>=1.2.0",)),
            "Old": _info(pip_extra=("dummy<1.2",), superseded=True),
        }
    )

    assert _expected_extras() == {"dummy": ["dummy>=1.2.0"]}


def test_non_superseded_pins_are_unioned(registry):
    """Without the flag the two pins are unioned, which is what makes the extra unresolvable."""
    registry(
        {
            "Current": _info(pip_extra=("dummy>=1.2.0",)),
            "Old": _info(pip_extra=("dummy<1.2",)),
        }
    )
    assert _expected_extras() == {"dummy": ["dummy<1.2", "dummy>=1.2.0"]}


def test_folder_with_only_superseded_entries_drops_out(registry):
    """A folder whose every entry is superseded contributes no extra at all."""
    registry({"Old": _info(pip_extra=("dummy<1.2",), superseded=True)})
    assert _expected_extras() == {}


def test_real_registry_keeps_tabdpt_installable():
    """The shipped registry must not union TabDPT's mutually exclusive pins."""
    assert _expected_extras()["tabdpt"] == ["tabdpt>=1.2.0"]
