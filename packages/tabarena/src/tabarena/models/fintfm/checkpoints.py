from __future__ import annotations

import hashlib
from typing import Any

#: A checkpoint reference is a plain dict, so it survives YAML and pickle round trips in a
#: config. A published one names a Hugging Face file by repo, filename, commit and SHA-256;
#: an unpublished one names the checkpoint and its SHA-256 only, and cannot be resolved.
CheckpointRef = dict[str, str]

_PUBLISHED_KEYS = frozenset({"repo_id", "filename", "revision", "sha256"})
_UNPUBLISHED_KEYS = frozenset({"unpublished", "sha256"})

#: The published binary checkpoint, Apache-2.0 and ungated, pinned to a **commit** rather than
#: a branch. A moving reference would let the weights change underneath a recorded leaderboard
#: number, which is the one thing a benchmark entry cannot tolerate.
BINARY_CHECKPOINT: CheckpointRef = {
    "repo_id": "kabartay/fintfm-binary",
    "filename": "v4-cellattn-labels.pt",
    "revision": "f116bfd43a2b15c65ed3551ea8c38e3364629ddc",
    "sha256": "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2",
}

#: Placeholder for the multiclass entry's checkpoint (``dr74-mc-s0``, ``max_classes=10``). Not
#: published; resolving it raises :class:`CheckpointNotPublishedError` until a published
#: reference replaces it. The SHA-256 identifies the file the entry was measured with.
MULTICLASS_CHECKPOINT_UNPUBLISHED: CheckpointRef = {
    "unpublished": "fintfm-multiclass-dr74-mc-s0",
    "sha256": "14d5b4d561fbea6e35c7a30dc2644cb8f64f437c25d62da087ee960717cc8f4d",
}

#: Placeholder for the regression entry's checkpoint (``v4-binned48_3-control``,
#: ``max_classes=10``). Not published; see :data:`MULTICLASS_CHECKPOINT_UNPUBLISHED`.
REGRESSION_CHECKPOINT_UNPUBLISHED: CheckpointRef = {
    "unpublished": "fintfm-regression-v4-binned48_3-control",
    "sha256": "31d5b3ef49e89bbe405fc85ab1d746a9fd6c3a1c5282e1770111902cc4af526c",
}


class CheckpointNotPublishedError(RuntimeError):
    """The entry's checkpoint for a problem type has no published location yet."""


def sha256_of(path: str) -> str:
    """SHA-256 hex digest of the file at ``path``, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_ref(ref: Any, *, source: str) -> None:
    """Raise ``ValueError`` unless ``ref`` is a local path string or a well-formed reference.

    Args:
        ref: The value of a ``checkpoint_<problem type>`` hyperparameter.
        source: Where the value came from, for the message.
    """
    if isinstance(ref, str):
        if not ref.strip():
            raise ValueError(f"{source}: the checkpoint path is empty")
        return
    if not isinstance(ref, dict):
        raise ValueError(f"{source}: a checkpoint is a local path or a reference dict, got {type(ref).__name__}")
    keys = set(ref)
    if keys not in (_PUBLISHED_KEYS, _UNPUBLISHED_KEYS):
        raise ValueError(
            f"{source}: checkpoint reference keys {sorted(keys)} match neither a published reference "
            f"{sorted(_PUBLISHED_KEYS)} nor an unpublished placeholder {sorted(_UNPUBLISHED_KEYS)}"
        )


def resolve_checkpoint(ref: Any, *, problem_type: str) -> str:
    """The local path of the checkpoint ``ref`` names, downloading and verifying it if published.

    A string is a local path and is returned unchanged (it comes from an environment variable
    or an explicit hyperparameter, never from an entry default). A published reference is
    fetched from the Hugging Face cache at its pinned commit and its SHA-256 is checked.

    Args:
        ref: A path string, or a reference dict from :func:`validate_checkpoint_ref`.
        problem_type: ``binary``, ``multiclass`` or ``regression``, for the error message.

    Returns:
        The path of a checkpoint file.

    Raises:
        CheckpointNotPublishedError: ``ref`` is an unpublished placeholder.
        ValueError: ``ref`` is malformed, or the downloaded file's SHA-256 differs.
    """
    validate_checkpoint_ref(ref, source=f"checkpoint_{problem_type}")
    if isinstance(ref, str):
        return ref
    if "unpublished" in ref:
        raise CheckpointNotPublishedError(
            f"The FinTFM {problem_type} checkpoint {ref['unpublished']!r} (sha256 {ref['sha256']}) has no "
            f"published location yet, so the entry cannot resolve it. Supply a local file with the "
            f"checkpoint_{problem_type} hyperparameter, or the FINTFM_CHECKPOINT_{problem_type.upper()} "
            f"environment variable."
        )
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=ref["repo_id"], filename=ref["filename"], revision=ref["revision"])
    actual = sha256_of(path)
    if actual != ref["sha256"]:
        raise ValueError(
            f"FinTFM {problem_type} checkpoint {ref['repo_id']}/{ref['filename']}@{ref['revision']} has "
            f"sha256 {actual}, expected {ref['sha256']}"
        )
    return path
