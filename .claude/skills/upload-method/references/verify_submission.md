# Verifying submitted results before hosting them

Most methods are hosted from a run we launched ourselves. Sometimes a contributor's own run is good enough to
host directly, which saves a maintainer re-run that can cost thousands of dollars for a model with a search
space. Before such results are processed and marked `verified=True`, check that they are what they claim to be.
This page is the procedure, first used on PR #681 (TabM-A and MLP-A, TabArena-Full, 2 x 164,016 results).

The checks below test the *files*. The *code* that produced them gets the add-model skill's agent review
(`.claude/skills/add-model/references/model_review.md`) first, on the PR head and on the pinned library.
Then confirm that the commit the contributor names as having produced the results carries the same model code
as the PR head (`git diff <results-commit> <pr-head> -- packages/tabarena/src/tabarena/models/<key>`).

## 1. Download and extract safely

- Download into its own empty directory under the workspace (`<workspace>/submissions/<pr>_<model>/download`),
  pinned to a revision (for Hugging Face, `hf download ... --revision <sha>`), and compare the checksums the
  contributor publishes (`sha256sum`).
- Extract into a second empty directory. Keep the scripts elsewhere and run Python with `-I`.
- A submitted `results.pkl` is untrusted: plain `pickle.load` executes code the file names. Load it only with
  `scripts/safe_load.py`, which resolves numpy and AutoGluon params-dict classes and refuses everything else.
  The scan below runs every file through it, so a clean scan also shows that no file references other code.
  Running the PR's own Python (its `hpo.py`, `info.py`) may be refused by the auto-mode classifier until the
  maintainer approves; re-implementing a short numpy search space is the fallback.

## 2. Scan every file

```bash
S=.claude/skills/upload-method/scripts
python -I $S/scan_submission.py <extracted>/data <scratch>/scan.parquet --procs 80
# split hashes of one of our own full runs (any method, one config with all splits):
python -I $S/scan_submission.py <workspace>/output/<our_run>/data/<Config>_c1_default_BAG_L1 <scratch>/ref.parquet --reference
python -I $S/summarize_submission.py <scratch>/scan.parquet --reference <scratch>/ref.parquet \
    --expected-configs <scratch>/expected.json --new-results <scratch>/new_results.parquet
```

`expected.json` maps each config directory name to the hyperparameters the PR's search space generates for it
(the `rN` configs of a `CustomAGConfigGenerator` come from its `search_space_func(num_random_configs, seed)`).

What to expect from a valid submission:

- Coverage: every config on every task and split of the claimed subset (TabArena-Full: 51 datasets, 816
  splits, 9 or 30 per dataset), `n_children` 8, the 3600 s limit, no `time_train_s` far above it.
  `run_process_method.py` (inspect) shows the same from the processing side, including `validation_protocol`.
- Splits and labels: `y_test_idx`, `y_val_idx`, `y_test`, `y_val` hash-identical to our own run on every split.
  A mismatch means other splits or altered labels and ends the review.
- Hyperparameters: every config identical to the search space's draw, and one variant per config across splits.
- Metrics: `metric_error` recomputed from the stored test predictions matches exactly for `roc_auc` and `rmse`.
  Multiclass `log_loss` differs by up to about 1% through probability clipping; that is benign.
- Predictions: finite, probabilities in [0, 1], rows summing to 1.
- Hidden failures: a wrapper that catches divergence can report a fit that is no better than a constant
  predictor (`constant_predictions`, `all_children_const`). TabArena requires every config to succeed, so these
  are failures even though nothing crashed. Group them by config and look up which search-space choice they share
  (on PR #681, one model module explained most of them). Datasets on which every method sits near the constant
  baseline (`hiva_agnostic`, scored as multiclass log-loss) inflate these counts; the c1 rows show that baseline.
- Leakage signals: the generalization gap `log(test / val)` should resemble that of the closest existing method.
  Where the new method beats a baseline, its test gain should not exceed its validation gain; test information
  in the predictions shows up as the opposite.

## 3. Reproduce the claimed leaderboard numbers

Default and tuned come straight from the scan (`--new-results`). Tuned + ensembled needs the predictions:

```bash
python -I $S/ensemble_check.py <extracted>/data <scratch>/ensemble.parquet --procs 80
```

Concatenate both tables (fill the missing columns of the ensemble rows from the tuned rows) and score them
against the arena from the repo root:

```python
from tabarena.contexts import TabArenaContext
lb = TabArenaContext().compare(output_dir=..., new_results=new_results, plot=False)
```

Check the method's rows against the claimed Elo. On PR #681 default and tuned matched the claims within 5 Elo,
and the approximate ensemble came within 1 Elo of the value the processed artifact later gave (the authors'
own claim was 10 higher). To compare per dataset with an existing method, load its per-config rows with
`TabArenaContext().load_model_results(methods=[...])` and pick the tuned config by `metric_error_val` the same
way; for TabM this reproduces the official default exactly and the tuned rows within 0.0014.

## 4. Estimate the impact of a problem before deciding

When the review finds a protocol problem confined to part of the search space (one module, a range of a
hyperparameter), rebuild tuned and tuned + ensembled without those configs (`ensemble_check.py --exclude ...`)
and score both versions with `compare`. Replace the method's rows instead of adding a second copy under the
same `config_type`, which `compare` collapses. On PR #681 dropping a transductive module (25 of 200 configs) moved
the Elo by 1.5, so the maintainers accepted the results as an exception.

## 5. Decide and report

The maintainer decides; give them the verdict with the numbers: what was checked and passed, each problem with
its scale (how many results, which configs, Elo impact), and what a future submission must change. Accepted
exceptions go into a code comment next to the affected code and into the review comment on the PR. With the
maintainer's sign-off, process and upload as in the main skill with `verified=True`, then remove the download
and extraction directories once r2 is verified (they replace `<workspace>/output/<benchmark_name>/` in Step 8).
