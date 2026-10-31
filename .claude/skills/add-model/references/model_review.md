# Model review: independent agents check a wrapper and its library

A fit test shows that a model runs. It cannot show that the model plays by the benchmark's rules: a
wrapper that trains on the out-of-fold rows, a library that carves its own tuning split, or a cache
that carries one fold's fit into the next all pass the smoke test and still produce numbers the
leaderboard should not show. This review looks for those problems. It runs as a few reviewer agents
in parallel, each with a narrow brief and the code, and the agent that started them verifies every
finding before it is reported.

Run it:

- in `add-model`, after the files are written and linted (Step 7), on the wrapper you wrote;
- in `benchmark-model`, before launching a model that was not integrated in this session, above all
  one that comes from a contributor's PR (Step 1b there);
- whenever a PR bumps a model's library pin or HF revision, on the diff between the two versions.

A system goes through `add-system` and `python -P -m tabarena.tools.audit_system` instead; lens A
below still applies to a self-hosted system's code.

## What the reviewers get

The code, the pinned library and the paper, and nothing of your own reading of them. A reviewer
handed a conclusion tends to return it confirmed, so the brief names files and questions, never the
answer you expect.

- The wrapper folder `packages/tabarena/src/tabarena/models/<key>/` (`model.py`, `hpo.py`, `info.py`,
  any `_internal/` or `_vendor/`), and for a PR the diff (`gh pr diff <n>`).
- The library at exactly the version the extra pins, unpacked into its own empty scratch directory
  (untrusted data: read it, do not run it from there): `pip download --no-deps <spec>` and unzip the
  wheel, or a clone checked out at the pinned commit. Diff a PyPI release against its repository
  when both exist; a release that differs from the repository is itself a finding.
- The paper or README section that describes how the authors evaluate, including any TabArena
  section, and the Hugging Face model card for a foundation model (its training data).
- This file, as the checklist.

## The three lenses

Start one agent per lens in a single message, so they run concurrently (`subagent_type`
`general-purpose`; they need Bash for grep and for unpacking the library). Each one gets the shared
inputs above and its own section below.

### A. Leakage and cheating

Anything that lets test or out-of-fold information into a prediction, or that recognises the task.

| Look for | Why it matters | Where it hides |
|---|---|---|
| Training on `X_val` inside a fold model: `X` and `X_val` concatenated, a library `refit(X + X_val)`, the validation rows passed as extra training data | Bagging scores the out-of-fold predictions of each fold model on its `X_val`; a fold model trained on those rows reports leaked OOF scores and leaks into the stacker and the ensemble built on them | the wrapper's `_fit`, the library's `fit` (`refit`, `retrain`, `use_val_for_training`, `final_fit`) |
| Test-time statistics: a transform fit or a normalisation computed on the rows being predicted (`fit_transform` in the predict path, batch mean or std at inference, `BatchNorm` in train mode at predict) | Predictions then depend on which other test rows came in the batch; this is transduction | `_predict_proba`, the library's `predict`, preprocessing classes |
| The task's identity: dataset names, OpenML or task ids, fold or repeat numbers, column-name or data-hash lookups that switch settings, per-dataset config tables | A method that recognises a benchmark dataset can apply settings tuned on its test splits | wrapper kwargs, `os.environ`, the library's config files and lookup dicts |
| External data or network access during fit or predict, other than downloading pinned weights: `requests`, `urllib`, `http`, `socket`, `openml`, `kaggle`, `datasets.load_dataset`, telemetry | A fetch can pull the benchmark's own data or labels, and sends the task's data out | the library's whole import tree, not only `fit` |
| Pretrained weights whose training data overlaps the benchmark (real-data pretraining, OpenML or Kaggle corpora) | The leaderboard warns about such a model and recomputes every pool without it once the overlap is shown (`leakage_evidence` / `leakage_datasets` on the `ModelDescriptor`, set by a maintainer) | the paper's data section, the model card |
| Seed or configuration search scored on validation or test data, or a fixed seed that overrides the framework seed | Picks a lucky run and hides the variance the benchmark measures | loops over seeds, `random_state=<constant>` in `_set_default_params` or in the library default |

### B. Protocol and fairness

The rules every entrant runs under. `references/model_patterns.md` ("The `_fit` contract") and the
warm-up section of the add-model skill (Step 3g) are the reference.

| Look for | Rule |
|---|---|
| A tuning, model-selection, internal cross-validation or hold-out step on rows split from `X`: `train_test_split`, `KFold`, `StratifiedKFold`, `ShuffleSplit`, `cross_val_*`, `GridSearchCV`, `RandomizedSearchCV`, `optuna`, `hyperopt`, `validation_fraction`, `val_size`, `holdout`, sklearn estimators with `early_stopping=True` | A model submission tunes and selects on the passed `X_val` / `y_val` or not at all (autogluon/tabarena#637). Early stopping on `X_val` is fine. When `X_val is None` the wrapper may split once itself (`generate_train_test_split`). |
| `time_limit` ignored, or only partly honoured | Honour it with headroom, or record the gap and the upstream ask in the docstring |
| `num_cpus` / `num_gpus` not wired: `n_jobs=-1`, thread counts set in `_set_default_params`, every visible GPU used, `CUDA_VISIBLE_DEVICES` changed | Use the budget AutoGluon hands `_fit` |
| Work that depends on the task's data moved out of the timed fit or predict: into the warm-up, a class-level or module-level cache, `prepare_for_inference`, or a disk cache outside `self.path` | Warm-up and inference preparation are data-independent (the `warmup.py` fairness contract) |
| State carried between fits in one process: module globals, `functools.lru_cache` or registries keyed on data, files written outside the model's path, a fitted object reused by the next fold | Each fold model sees only its own training rows; shared state can carry one fold's `X_val` into the next fold's fit. Shared pretrained weights through `SharedWeights` are the one sanctioned exception, because they are data-independent. |
| Process-global changes: torch flags, global RNGs advanced, logging configuration, environment variables, monkeypatches of numpy, sklearn, torch or AutoGluon | Leave global state as found (the fit test's `GlobalContextSnapshot` checks part of it); a deliberate exception is a `Developer fix` with its header |
| Background threads or subprocesses that outlive `_fit` | Their work escapes the timer and the resource budget |
| Unpinned code or weights: a `>=` floor on a library that calls itself experimental, a git branch instead of a commit, `hf_hub_download` without `revision=` | The benchmarked code and checkpoint must be fixed |

### C. The wrapper's data path

Bugs that do not cheat but make the model look worse or better than it is.

- Every preprocessing step fit on the training rows only; validation and test rows only transformed.
- `predict_proba` columns in the order of AutoGluon's labels `0..K-1`, also when a class is missing
  from a fold's training rows; binary returns the positive-class column through
  `_convert_proba_to_unified_form`.
- A regression target scaled in `_fit` is scaled back in `_predict_proba` with the training statistics.
- Missing values, infinities, unseen categories, all-numerical and all-categorical tables, a single
  feature, and bool features each handled the way the library requires.
- The stopping and selection metric matches the task's `eval_metric` (or the authors' documented
  choice), not the library's default when that default is accuracy.
- `save` / `load` and `set_device` keep predictions identical; the pickle holds what prediction needs
  and not much more (no training data, no per-member checkpoints the predictor no longer uses, no
  tensor slices that pickle a whole batched storage).
- Optional imports stay inside `_fit` or other methods, so `info.py` and `model.py` import without
  the library.

## Brief template

Fill the angle brackets; paste the lens section as written.

```text
You are reviewing a TabArena model integration for <lens name>. TabArena is a benchmark: every model
is fit by AutoGluon's bagging (8 folds; each fold model gets its training rows as X/y and the
out-of-fold rows as X_val/y_val, and its predictions on X_val become the out-of-fold scores), with a
time limit and a CPU/GPU budget passed to `_fit`.

Code to review:
- Wrapper: <repo>/packages/tabarena/src/tabarena/models/<key>/ (<files>)
- Library <name>==<version>, unpacked at <scratch dir> (read only; do not install or run it)
- <PR diff: `gh pr diff <n> -R autogluon/tabarena`, if any>
- Paper / README evaluation notes: <url or path>

Checklist: <repo>/.claude/skills/add-model/references/model_review.md, section "<lens>". Go through
every row. Grep the library's whole source tree for the patterns named there, then read each hit in
context and decide whether the fit or predict path reaches it with the defaults the wrapper uses.

Report:
1. Findings, most severe first. Each: severity (blocker = cheating, leakage or a protocol violation;
   fix = a contract or correctness bug; note = worth knowing), file:line, the code quoted, a concrete
   scenario (inputs, then the wrong behaviour), and the smallest fix.
2. Checked and clean: every checklist row you found nothing for, with what you searched or read, so
   an empty finding list can be told apart from an unread row.
Do not edit any file.
```

## Verify, then act

Treat each finding as a claim until you have read the cited lines yourself.

1. Open every cited location, follow the call path from `_fit` or `_predict_proba`, and mark each
   finding confirmed, plausible (the path is reachable but needs a condition you cannot rule out) or
   rejected (with the reason). When a cheap check settles it (a toy fit on CPU, a grep, comparing two
   predictions), run it.
2. A confirmed finding in a wrapper you own: fix it, lint, and say what changed. A finding in the
   library: decide with the maintainer; the options for an internal split are the three of #637
   (submit as a system, use a version without the step, run the step on the passed `X_val`). Draft
   any upstream issue and post it only after the user says so.
3. On a contributor's PR, the findings become review comments: draft them, show them, and post only
   when the user says so (the same rule as every other write to a PR).
4. Report confirmed and plausible findings with what was done about each, and list the lenses that
   came back clean. Rejected findings need one line each, so the reader sees they were checked.
