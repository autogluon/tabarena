<!-- Keep the body short. Reviewers read the summary and open Details only when they need to. -->

## Summary

<!-- Two to four sentences: what changed and why. -->

<details>
<summary>Details</summary>

<!-- The longer description for reviewers: design decisions, alternatives considered, how to verify the change. Delete this block if the summary says it all. -->

</details>

## Tests

<!-- The commands you ran, e.g. `ruff check . && ruff format --check .` and `pytest tests/tabarena/<area> -q`. -->

## Model or system submission

<!-- Delete this section if the PR is not a model or system submission. -->

TabArena is not a benchmarking service. We accept methods their authors have already evaluated with the official pipeline and confirm the results by re-running them. See [Contributing a Model or System](https://github.com/autogluon/tabarena#contributing-a-model-or-system).

A submission reaches the leaderboard in five stages. This pull request is stage 1; maintainers drive the other four and post in the PR as each one completes, so this thread is the record of the submission.

1. Pull request with the method. You integrate the model or system, run the quick start and the TabArena-Lite evaluation, and open this PR with the checklists below filled in. Reviewers check the wrapper against the protocol (the validation split TabArena passes, time limits, resources, seeds), the dependency pin and license, and the Lite results, and help fix what the review finds.
2. Benchmark run. Once the review is through, a maintainer runs the method on the full TabArena task set on the benchmark hardware: all splits of every dataset, and for a model the default configuration plus the full search space. Every entrant is measured this way, so your run from stage 1 is never the final entry. The run is recorded in `packages/tabflow_slurm/BENCHMARK_LOG.md` and the maintainer posts the resulting leaderboard position here.
3. Verification and sign-off. Maintainers compare the run with the Lite results from stage 1 and look into failed splits and time-limit hits; a mismatch goes back to stage 1 or 2 with a fix. Then the person named under "Results" below confirms in the PR that the run reflects the method as intended (the right version, checkpoint and hyperparameters). That confirmation marks the entry as verified on the leaderboard. Without it the entry is still listed, but as unverified.
4. Upload and merge. The maintainer processes the raw results, hosts the artifacts (results, processed predictions and raw predictions), records the hosted locations and the verification status in the method's `info.py`, registers the method in the arena's method collection, and merges the PR. A merged method therefore always has hosted results behind it.
5. Leaderboard update. After the merge, the website artifacts are regenerated from the hosted results and the leaderboard is refreshed with a new version entry that names the method. This happens in batches with other new entries, usually within days of the merge.

Kind: <!-- model or system; if it replaces an entrant already on the leaderboard, name it -->

**Files**

- [ ] Model: `models/<key>/{__init__,model,hpo,info}.py`, a lazy entry in `models/__init__.py`, the extra in `packages/tabarena/pyproject.toml` (also listed in `extended`), the family in `website/website_format.py`. A `tests/tabarena/models/smoke_configs.py` override only if the toy fit needs it; no per-model test file and no edits to shared test infrastructure.
- [ ] System: `systems/<key>/{__init__,system,hpo,info}.py` with `MethodMetadata.system(...)` and `tags` chosen from `with-llm` / `closed-source-api`, plus the extra in `packages/tabarena/pyproject.toml`.

**Implementation**

- [ ] Model: `_fit` uses the `X_val` / `y_val` that TabArena passes (no internal cross-validation), and the search space in `hpo.py` supports about 200 configurations. Variants are search-space parameters, not separate models.
- [ ] `time_limit`, `num_cpus` / `num_gpus` and the seed are wired through; nothing is monkey-patched globally; optional imports stay inside the wrapper; the model pickles for parallel bagging.
- [ ] The dependency is pip-installable with an exact pin (PyPI version or git commit, no vendored code) and the extra matches `pip_extra` in `info.py`; Hugging Face checkpoints pin a revision; maintainers can run it without credentials or gated weights; the license, the paper and documentation links, and the supported problem types are stated.
- [ ] `ruff check` and `ruff format --check` pass. Model: `pytest -m models -k <Key>` passes. System: `pytest tests/tabarena/systems/` passes and the system quick start ran.

**Results** (we need these before we re-run the method)

- [ ] TabArena-Lite (`subset="lite"`) with the official pipeline: the default plus about 25 random HPO configurations where the method has a search space; or the BeyondArena `core` subset.
- [ ] The hardware (CPU, GPU, RAM) and the entry-point script are stated below.
- [ ] Optional: a link to the run's output directory (the `expname` folder with the `results.pkl` files, zipped or as a GitHub / Hugging Face release) so we can verify and integrate the results directly.
- [ ] Who signs off on the maintainer re-run on behalf of the authors: <!-- GitHub handle -->

<!-- Paste the leaderboard table printed by the quick start, the hardware, and the command you ran. -->

---

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.
