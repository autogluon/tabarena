"""How to configure *where* TabArena caches everything — one object, set once.

TabArena reads and writes five independent caches. By default they live under your home
directory, which is usually the wrong disk for benchmarking (model weights and datasets are
large, and on a cluster every worker needs to see the same files). Instead of exporting a
handful of environment variables on every machine, declare the locations once with a
``CacheConfig`` and hand it to the context — it points the driver at them immediately and
re-applies them on every worker that runs a job.

The five caches (see ``tabarena.caching.CacheConfig`` for the authoritative reference):

  * ``openml``       — THE important one. The materialized datasets + their cross-validation
                       splits, plus all TabArena-derived task artifacts, live here. Point this
                       at a large, ideally shared, disk.
  * ``huggingface``  — foundation-model weights (TabPFN / Mitra / LimiX / ...). Sets ``HF_HOME``.
  * ``data_foundry`` — the one-time raw dataset download for data_foundry / BeyondArena (which is
                       then converted into the OpenML cache). Sets ``DATA_FOUNDRY_CACHE``. NOTE:
                       this is NOT ``HF_HOME`` — data_foundry passes an explicit cache dir to
                       ``snapshot_download``, so its downloads do not follow ``HF_HOME``.
  * ``tabarena``     — TabArena's own results / baselines / leaderboard artifacts.
  * ``results``      — the per-run output cache (the runner's ``expname``). Optional; if unset a
                       throwaway temp dir is used unless you pass ``expname=`` explicitly.

Run it:

    python examples/benchmarking/run_configure_caches.py
"""

from __future__ import annotations

from tabarena.caching import CacheConfig
from tabarena.contexts import TabArenaContext

if __name__ == "__main__":
    # Option A — put every cache under one parent directory (the common case: one big disk).
    #   /data/tabarena-caches/openml, /huggingface, /data_foundry, /tabarena, /results
    cache_config = CacheConfig.from_root("/data/tabarena-caches")

    # Option B — set each location explicitly (e.g. datasets on shared storage, weights local).
    #   Any field left out stays at its current env var / library default.
    cache_config = CacheConfig(
        openml="/shared/openml-cache",
        huggingface="/shared/hf-cache",
        data_foundry="/shared/data-foundry-cache",
        tabarena="/shared/tabarena-cache",
        results="/scratch/tabarena-results",
    )

    # Hand it to the context once. The context calls cache_config.apply() now (configuring THIS
    # process) and again inside run_jobs (so any distributed worker that runs the context inherits
    # the same locations) — no per-worker setup, no environment-variable juggling.
    context = TabArenaContext(cache_config=cache_config)

    # From here, everything — build_jobs(pre_materialize=...), run_jobs / build_and_run_jobs,
    # compare — reads and writes the directories you declared above. For example:
    #
    #     experiments = TabArenaV0pt1ExperimentBundle(models=[("Linear", 0)]).build_experiments()
    #     context.build_and_run_jobs(experiments, expname=None, subset="lite")  # expname falls
    #                                                                            # back to results
    #
    # You can also apply a CacheConfig directly in any process you control (e.g. a custom
    # distributed worker's entrypoint) without a context:
    #
    #     CacheConfig.from_root("/data/tabarena-caches").apply()
    #
    # If you already use OpenML elsewhere and don't want TabArena to permanently change its
    # global cache location, set scope_openml=True on the CacheConfig: TabArena points OpenML at
    # cache_config.openml only for the duration of each run and restores your previous
    # openml.config afterwards. (apply_on_run=False is the other policy flag — it stops the
    # context re-applying the config inside run_jobs.)
    #
    #     cache_config = CacheConfig(openml="/shared/openml-cache", scope_openml=True)
    #     context = TabArenaContext(cache_config=cache_config)
    print("Configured caches for context:", context.cache_config)
