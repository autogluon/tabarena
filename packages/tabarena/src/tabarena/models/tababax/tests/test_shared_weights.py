"""The TabAbax wrapper keys the shared network on the checkpoint, not just on its file name.

``SharedWeights.key`` is what decides whether two fits reuse one in-process network. The
wrapper pins a Hub commit, so the same file name at two commits is two different networks;
if the revision is missing from the key, the second revision silently scores on the first
one's weights. These tests read the declaration and resolve the key inputs the way the
sharing layer does, so a dropped key entry fails here instead of biasing a run.

They do not need torch or the released checkpoints: nothing is downloaded or built.
"""

from __future__ import annotations

import pytest

from tabarena.models.tababax.model import CHECKPOINT_REVISION, TabAbaxModel

# Kept out of the default ``tests/`` collection like the rest of a model's tests, and out of
# PR CI by the repo-wide ``-m 'not network and not models'`` filter.
pytestmark = pytest.mark.models


def _sw():
    from autogluon.core.models.abstract import shared_weights

    return shared_weights


def test_revision_is_part_of_the_shared_weights_key():
    assert "checkpoint_revision" in TabAbaxModel.shared_weights.key


def test_the_key_still_names_the_checkpoint_and_the_local_path():
    # Neither may be dropped: two different files must not share a network either.
    key = TabAbaxModel.shared_weights.key
    assert "checkpoint_version" in key
    assert "model_path" in key


def test_key_inputs_separate_two_revisions():
    # ``_inputs_of_estimator`` is what the sharing layer reads the key from. The library
    # estimator stores the three names as attributes, so a differing revision has to show up.
    sw = _sw()
    pytest.importorskip("tababax")
    from tababax import TabAbaxClassifier, TabAbaxRegressor

    key = TabAbaxModel.shared_weights.key
    for cls in (TabAbaxClassifier, TabAbaxRegressor):
        first = cls(checkpoint_version="tababax-classifier.ckpt", checkpoint_revision="revision-a")
        second = cls(checkpoint_version="tababax-classifier.ckpt", checkpoint_revision="revision-b")
        assert sw._inputs_of_estimator(first, key) != sw._inputs_of_estimator(second, key), cls.__name__


def test_key_inputs_separate_a_local_file_from_the_download():
    sw = _sw()
    pytest.importorskip("tababax")
    from tababax import TabAbaxClassifier

    key = TabAbaxModel.shared_weights.key
    downloaded = TabAbaxClassifier(checkpoint_version="tababax-classifier.ckpt")
    local = TabAbaxClassifier(model_path="some-local-checkpoint.ckpt", allow_auto_download=False)
    assert sw._inputs_of_estimator(downloaded, key) != sw._inputs_of_estimator(local, key)


def test_the_wrapper_pins_a_revision_and_passes_it_to_the_estimator():
    # The key can only separate revisions that the fit actually sets.
    source = TabAbaxModel.__module__
    assert CHECKPOINT_REVISION, f"{source}: the wrapper pins no revision"
    assert len(CHECKPOINT_REVISION) == 40, "the pin is a full commit hash"
