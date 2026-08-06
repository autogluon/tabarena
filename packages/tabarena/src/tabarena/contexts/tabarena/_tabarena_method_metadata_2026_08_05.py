"""Method metadata for the ``tabarena-2026-08-05`` suite: the released AutoGluon 1.6 presets.

The AutoGluon 1.6 release runs on TabArena-Full (816 tasks over 51 datasets), fit with the shipped
presets and a 4 hour per-task time limit. ``compute`` is set manually: the runs were on one GPU per
task, but the raw results record ``num_gpus=0``, so the inferred value would be wrong.
"""

from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata

_common_kwargs = dict(
    suite="tabarena-2026-08-05",
    date="2026-08-05",
    compute="gpu",
    date_introduced="2026-08",  # AutoGluon 1.6 release
    reference_url="https://arxiv.org/abs/2003.06505",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

ag_160_eq_4h_metadata = MethodMetadata.baseline(
    method="AutoGluon_16_extreme",
    name="AutoGluon 1.6 (extreme, 4h)",
    **_common_kwargs,
)

# The `noncommercial` preset: `extreme_quality` plus TabPFN-3, whose license is not commercially
# permissive; the entry itself carries no license flag (MethodMetadata has none), the name does.
ag_160_noncomm_4h_metadata = MethodMetadata.baseline(
    method="AutoGluon_16_noncommercial",
    name="AutoGluon 1.6 (noncommercial, 4h)",
    **_common_kwargs,
)

methods_2026_08_05_ag: list[MethodMetadata] = [
    ag_160_eq_4h_metadata,
    ag_160_noncomm_4h_metadata,
]
