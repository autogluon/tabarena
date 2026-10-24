from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.ftfm.hpo import gen_ftfm
from tabarena.models.ftfm.model import FTFMModel

ftfm_descriptor = ModelDescriptor(
    display_name="FTFM",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/machinelearningnuremberg/ftfm-official",
    organization="University of Technology Nuremberg",
    commercial_use=False,
    license="CC-BY-NC-4.0",  # the weights; the inference code is Apache-2.0
    date_introduced="2026-10-03",  # first public release of the code and weights
)

# Pretraining mixes synthetic tables with TabDPT's OpenML corpus (stages B and C, arXiv 2610.25558 App. F);
# the paper removes the tables overlapping TabArena or BeyondArena (123 -> 108) without saying how overlap
# was matched, and the library's ensemble defaults were chosen on TabArena tables (comments in
# ftfm/ensemble.py). Recheck both before comparing against a future version or a new arena.
# Run ftfmrerun_21102026 (SkyPilot pool, 32 RTX PRO 6000, BENCHMARK_LOG.md): the 594 classification splits, default config;
# verified (maintainer sign-off 2026-10-21).
ftfm_method_metadata = ftfm_descriptor.method_metadata(
    method="FTFM",
    ag_key="TA-FTFM",
    model_key="FTFM",
    can_hpo=False,
    date="2026-10-21",
    verified=True,
    validation_protocol="8x1",
    suite="tabarena-2026-10-21",
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

ftfm_info = ModelInfo(
    model_cls=FTFMModel,
    search_space=gen_ftfm,
    method_metadata=ftfm_method_metadata,
    # The inference package is not on PyPI; pinned to the commit the wrapper was written against.
    pip_extra=(
        "ftfm @ git+https://github.com/machinelearningnuremberg/ftfm-official.git@7e5e237daa26ef20f0faec1650d000b04de5122e",
    ),
    prefetch_weights=FTFMModel.prefetch_weights,
)
