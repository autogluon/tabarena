---
name: adapt-tabarena
description: Adapt TabArena/bencheval as the models/metrics/splitting layer for a new, domain-specific benchmark that lives in its own repository (not a contribution to this repo). Use this skill whenever the user wants to build a benchmark for a different data domain (e.g. spectroscopy, genomics, time series, a vertical-specific tabular task) on top of TabArena's model zoo and bencheval's leaderboard math, rather than reimplementing that layer from scratch. Triggers on "build a benchmark using TabArena", "depend on TabArena for our own benchmark", "port our benchmark onto TabArena/bencheval", "how do we reuse TabArena's models for X". Covers what to depend on vs. reimplement, the repeated k-fold + group-aware splitting protocol, layering domain preprocessing without forking, bagging parity, and the git-dependency PyPI trap downstream packages hit. Complements `add-model` (for contributing a model back into *this* repo) — this skill is for *consuming* TabArena from an external repo.
argument-hint: <YourBenchmarkName> [<domain>]
user-invocable: true
---

# Adapt TabArena for a New Domain Benchmark

This skill is for a **different repository** building its own benchmark on top of TabArena, not for contributing to this repo. If the user wants to add a model, system, or feature to TabArena itself, use `add-model` / `add-system` instead.

The concrete reference implementation this skill distills is **RamanBench**
(`github.com/ml-lab-htw/RamanBench`), which migrated its model/metrics/splitting layer
onto `tabarena`/`bencheval` directly for its v1 release, replacing a set of
hand-rolled patterns that were "inspired by" TabArena with the real thing. Read that
repo's `src/raman_bench/` if you want a worked example alongside this skill.

## Step 0: Scope the domain benchmark

Ask (or infer from context) what the downstream benchmark actually needs:

| Question | Why it matters |
|---|---|
| Does it need the full model zoo, or just leaderboard math over its own results? | Decides whether to depend on `tabarena` (models + HPO + splitting utilities) or only `bencheval` (Elo/win-rates/ranks/improvability from a results DataFrame — no dependency on `tabarena` at all). |
| Do the domain's datasets have replicate/group structure (multiple measurements per physical sample/subject)? | Decides whether the splitting layer needs group-awareness (Step 3) — a common but easy-to-miss leakage source outside plain tabular data. |
| Are there domain-specific models to run alongside TabArena's zoo (e.g. signal-specific architectures)? | Decides whether the downstream repo needs its own model registry layered next to TabArena's (Step 2). |
| Will the package be published to PyPI? | Decides whether the git-dependency isolation in Step 6 is needed now or can wait. |

## Step 1: Depend on the right package(s)

- **`bencheval`** — standalone, lightweight. Computes leaderboards (Elo, win-rates, ranks, improvability) from a results DataFrame you already have. No dependency on `tabarena`. Pull this in alone if the domain benchmark has its own models/splitting and only wants TabArena-grade leaderboard math.
- **`tabarena`** — the model registry, `ConfigGenerator`s, HPO search spaces, and splitting utilities (`tabarena.splits`, `tabarena.nips2025_utils.fetch_metadata`). Depends on `bencheval` and AutoGluon. Pull this in when the domain benchmark wants to reuse TabArena's existing model wrappers (classical ML, tabular DL, tabular foundation models) rather than reimplementing them.

Both currently install via git URL (`git+https://github.com/autogluon/tabarena.git#subdirectory=packages/<name>`), not a real PyPI release — see Step 6 before publishing the downstream package.

## Step 2: Model registry — reuse vs. layer your own

Do not fork TabArena's model wrappers. Two registries can coexist:

- **TabArena's own registry** (`tabarena.models.registry`) supplies the general-purpose tabular model zoo (classical ML through tabular foundation models) via `ConfigGenerator`. Use these as-is for anything not specific to the new domain.
- **A domain-specific registry**, layered in the downstream repo, holds only the models that don't belong upstream (a domain-specific architecture, a wrapper around a domain library). Build these the same way TabArena builds its own — an AutoGluon `AbstractModel`/`AbstractTorchModel` subclass with a `ConfigGenerator` for HPO — so they compose with the same fitting/splitting/bagging machinery as everything pulled from TabArena's registry. Do not invent a parallel fitting protocol for just the domain-specific models; the whole point of depending on TabArena is one shared protocol for every model regardless of where it's registered.

If a domain-specific model turns out to be broadly useful outside this one domain, that's a signal it belongs upstream instead — point the user at `add-model` in that case.

## Step 3: Splitting protocol — real repeated k-fold, adaptive repeats, groups

Reuse TabArena's actual splitting logic rather than reimplementing "inspired by" versions of it:

- **Repeated k-fold, not single holdout.** `RepeatedStratifiedKFold`/`RepeatedKFold` for ungrouped data. sklearn has no repeated wrapper for grouped data, so a domain benchmark with replicate structure needs a manually-repeated `StratifiedGroupKFold`/`GroupKFold` loop (repeat the fold assignment with a different `random_state` each pass, same as the ungrouped repeated splitters do internally).
- **Size-adaptive repeat counts, not a fixed number for every dataset.** Port `tabarena.nips2025_utils.fetch_metadata._get_n_repeats` (or the equivalent in the installed version) rather than picking one repeat count for the whole suite — TabArena's own curated metadata varies repeats by dataset size (more repeats for small datasets, where a single split is noisier) while holding fold count fixed. Verify the ported logic against TabArena's real per-dataset metadata, not just the docstring, before trusting it.
- **Group-awareness is not optional for domains with replicate measurements.** If the same physical sample/subject/specimen contributes multiple rows (common outside plain tabular data — repeated scans, multiple measurements per patient, technical replicates), every split must keep a group's rows together. Audit every dataset in the suite for this *before* the first real benchmark run: a dataset wrongly treated as row-independent silently leaks information from train into test and inflates every model's score, RamanBench's own experience being a concrete case of catching this late on datasets that had shipped without it.

## Step 4: Preprocessing — layer a mixin, don't fork

If the domain needs its own tunable preprocessing (denoising, domain-specific normalization, feature extraction), implement it as a mixin on top of AutoGluon's model classes, jointly tunable through the same HPO mechanism TabArena's own models use — not a separate preprocessing pass bolted in front of the pipeline. Key properties worth carrying over from a working implementation of this pattern:

- Each tunable step gets its own enable flag and hyperparameter search space, composed through one restriction dict so a benchmark config can turn steps on/off per run.
- Stateful steps (anything that fits parameters on training data, e.g. a reference-spectrum correction) must fit once on training data only and reuse that fit at transform time — never refit on the data being scored, in every fold.
- If a step changes the feature count or column names, resync whatever internal feature/column bookkeeping the underlying model class relies on after the transform runs — a preprocessing step that changes shape silently breaks any model wrapper that snapshots `X.columns` before your mixin's `_fit` runs.

## Step 5: Bagging and compute-budget parity

Match TabArena's own bagging default (`num_bag_folds=8`, from `AGModelBagExperiment`) rather than leaving it to whatever preset default AutoGluon happens to resolve to — an implicit, data-size-dependent bagging behavior is not a fair, reproducible protocol and makes historical vs. new results incomparable in ways that are hard to detect after the fact. If an earlier version of the domain benchmark ran without explicit bagging control, say so plainly in the new benchmark's changelog rather than presenting old and new numbers as directly comparable.

## Step 6: Packaging — the git-dependency PyPI trap

`tabarena` and `bencheval` install via git URL, and **PyPI rejects packages that declare a direct git dependency** in any extra (`Can't have direct dependency: ... ; extra == "..."`). If the downstream benchmark package is meant to publish to PyPI:

- Isolate the git-based `tabarena`/`bencheval` dependencies into their own optional-dependency extra (e.g. `[benchmark]`), separate from whatever extra holds only PyPI-installable packages.
- The base package and any PyPI-clean extras (e.g. an `[autogluon]` extra with just public AutoGluon packages) publish fine on their own.
- Full-functionality installs (anything needing the git-pinned extra) stay a from-source install (`pip install -e .[full]`) — document this split plainly rather than letting a release silently fail the PyPI publish step.
- Also check `pip index versions tabarena` before assuming a version pin resolves against the real package — a same-named placeholder can exist on PyPI (currently a `0.0.0` reserved release from the TabArena team itself, not a real release) that a bare `tabarena>=X` requirement would resolve against instead of the git-pinned dependency.

## Step 7: Cluster submission (optional)

If the domain benchmark runs on a SLURM cluster, the submission/resource-resolution tooling does not need to be domain-specific. A generic, profile-driven submit script (institution-specific values in a separate, git-ignored or private profile file; the submission logic itself public and reusable) lets the domain benchmark scale the same job matrix (`model x dataset x target x repeat x fold`) across whatever cluster capacity is available, and lets external contributors without cluster access still run smaller jobs locally against the same tooling.

## Step 8: Report

Summarize for the user:
- Which package(s) the downstream benchmark now depends on (`tabarena`, `bencheval`, or both) and why
- Where the domain-specific model registry lives relative to TabArena's own
- What changed in the splitting protocol (repeated k-fold, adaptive repeats, group-awareness) and which datasets needed a group-structure audit
- Whether the PyPI packaging split (Step 6) is needed now or deferred
- Any historical-vs-new comparability caveat introduced by protocol changes (Step 5), so it lands in the domain benchmark's own changelog
