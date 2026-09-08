---
name: adapt-tabarena
description: Adapt TabArena/bencheval as the tasks/models/metrics/splitting layer for a new, domain-specific benchmark that lives in its own repository (not a contribution to this repo). Use this skill whenever the user wants to build a benchmark for a different data domain (e.g. spectroscopy, genomics, time series, a vertical-specific tabular task) on top of TabArena's model zoo, task and experiment runner, and bencheval's leaderboard math, rather than reimplementing that layer from scratch. Triggers on "build a benchmark using TabArena", "depend on TabArena for our own benchmark", "port our benchmark onto TabArena/bencheval", "how do we reuse TabArena's models for X", "make a RamanBench-style arena for Y". Covers what to depend on vs. reimplement, turning your datasets into `UserTask`s, the repeated k-fold + group-aware splitting protocol (outer and inner), registering domain models next to TabArena's registry, layering domain preprocessing without forking, bagging parity, your own arena context and dataset catalogue, imputation and Elo anchoring, hosting results, the git-dependency PyPI trap, and what to report back to the TabArena maintainers so the API grows the hooks your benchmark needs. Complements `add-model` / `add-system` (for contributing back into *this* repo); this skill is for *consuming* TabArena from an external repo.
argument-hint: <YourBenchmarkName> [<domain>]
user-invocable: true
---

# Adapt TabArena for a New Domain Benchmark

This skill is for a **different repository** building its own benchmark on top of TabArena, not for
contributing to this repo. If the user wants to add a model, system, or feature to TabArena itself,
use `add-model` / `add-system` instead.

The concrete reference implementation this skill distills is **RamanBench**
(`github.com/ml-lab-htw/RamanBench`), which migrated its model/metrics/splitting layer onto
`tabarena`/`bencheval` for its v1 release, replacing hand-rolled patterns that were "inspired by"
TabArena with the real thing. Read that repo's `src/raman_bench/` for a worked example alongside
this skill.

Every module path below was checked against this checkout. Both packages are pre-1.0 and the API
still moves (several paths cited by earlier drafts of this skill no longer exist), so `grep` a path
before writing it into the downstream repo, and read the docstring of anything you depend on.

## What you reuse and what you write

| Layer | Reuse from `tabarena` / `bencheval` | You write |
|---|---|---|
| Datasets and outer splits | `UserTask`, `TaskMetadataCollection`, `TaskMetadataSource`, `SubsetPredicate` | dataset loaders, the split construction, a committed metadata CSV |
| Models and search spaces | the model registry, `ConfigGenerator`, every model's `hpo.py` | domain models registered via `register_model_info` |
| Fitting protocol | the experiment bundles, `AGModelBagExperiment`, the AutoGluon wrappers, the task-aware validation protocol | a bundle subclass carrying your defaults |
| Preprocessing | `TabArenaModelAgnosticPreprocessing`, `build_feature_generator`, model-specific hyperparameter injection | a domain feature generator or a model mixin |
| Running | `AbstractArenaContext.build_and_run_jobs`, `ExperimentBatchRunner`, `JobBatch`, `tabflow_slurm` | a cluster profile |
| Leaderboard math | `bencheval.evaluator.BenchmarkEvaluator`, reached through `compare` | nothing, or one `LeaderboardMetric` |
| Publishing | `MethodMetadata` artifact tiers, `format_leaderboard`, the interactive HTML explorers | a `methods.py` roster and a place to host results |

The canonical downstream loop, from `examples/benchmarking/run_quickstart_tabarena_custom_datasets.py`:

```python
from tabarena.benchmark.experiment import TabArenaExperimentBundle
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.contexts import AbstractArenaContext

tasks = [make_user_task(ds) for ds in my_datasets]            # Step 2
collection = TaskMetadataCollection.from_user_tasks(tasks)     # your suite
experiments = TabArenaExperimentBundle(                        # Step 5
    models=[("LightGBM", 0), ("RandomForest", 0), (my_gen, 10)],
    n_random_configs=50, preprocessing_pipelines=["default"],
).build_experiments(time_limit=3600)
context = AbstractArenaContext(task_metadata=collection, methods=[])   # no TabArena baselines
context.build_and_run_jobs(experiments, expname=results_dir, user_tasks=tasks, debug_mode=True)
leaderboard = context.compare(output_dir=eval_dir)             # Step 9
```

Everything below explains the decisions hidden in those ten lines.

## Step 0: Scope the domain benchmark

Ask (or infer from context) what the downstream benchmark actually needs:

| Question | Why it matters |
|---|---|
| Full model zoo, or only leaderboard math over results you already have? | `tabarena` (models, runner, splitting) vs. `bencheval` alone (Elo, win-rates, ranks, improvability from a results DataFrame; no `tabarena` dependency). |
| Do datasets have replicate or group structure (several rows per specimen, patient, sample)? | Both the outer splits (Step 3) and the inner bagging folds (Step 4) must be group-aware, and the inner ones need the `data-foundry` extra. |
| Do datasets have temporal structure? | Outer splits must be forward-in-time; TabArena's inner protocol handles `time_on` but not `time_on` together with `group_on`. |
| Do features arrive as thousands of numeric columns (spectra, expression arrays, embeddings)? | Foundation models carry feature and row caps; `ModelConstraints` (Step 5) skips incompatible pairs instead of failing them, and dimensionality reduction becomes a preprocessing decision (Step 7). |
| Do the domain's metrics differ from `roc_auc` / `log_loss` / `rmse`? | Any metric must be registered with AutoGluon by name before a task can reference it (Step 2). |
| Are there domain-specific models or whole pipelines? | Decides between a registered model (Step 6) and an `ExternalSystemModel`. |
| Will you host results so others can compare against them without rerunning? | Decides whether you need a `methods.py` roster, `MethodMetadata` storage config, and your own arena context (Step 10). |
| Cluster? PyPI release? | Steps 8 and 11. |

## Step 1: Depend on the right package(s)

`bencheval` is standalone and light (numpy, pandas, scipy, scikit-learn). It computes leaderboards
from a results DataFrame you already have. Pull it in alone if the domain benchmark keeps its own
models and splitting and only wants TabArena-grade leaderboard math. Its `__init__` is empty, so
import from submodules: `from bencheval.evaluator import BenchmarkEvaluator`.

`tabarena` adds the model registry, config generators and search spaces, the task and experiment
runner, the contexts, plotting, and the artifact tiers. It depends on `bencheval` and on an
AutoGluon pre-release. Extras that matter downstream: `[benchmark]` for model fitting (the core
model set plus `plot`, `text`, `preprocessing`, `data-foundry`), `[data-foundry]` on its own if you
only need grouped inner splits (see Step 4), `[plot]` for the figures `compare` renders, and one
extra per extended model (`[tabm]`, `[realmlp]`, `[tabpfn]`, ...). The registry's `pip_extra` on
each `ModelInfo` names the extra a model needs.

Both packages publish PyPI pre-releases at one shared version, and every `tabarena` release pins
`bencheval==<same version>`. Install with `pip install --pre tabarena` (or `uv pip install
--prerelease=allow`), and pin the pair together in the downstream `pyproject.toml`. See Step 11
before choosing between the PyPI release and a git URL.

## Step 2: Turn each dataset into a `UserTask`

`tabarena.benchmark.task.UserTask` is the local, OpenML-free task type. Its `create_task` computes
the task's `TabArenaTaskMetadata` (problem type, sizes, dtype flags, per-split statistics) and
`save_task` pickles dataset, splits and metadata to one file. Reference implementations:
`examples/benchmarking/run_quickstart_tabarena_custom_datasets.py` (plain DataFrames),
`tabarena/benchmark/task/data_foundry/adapter.py::convert_curated_container_to_user_task` (the
converter BeyondArena uses, a complete "third-party dataset object to `UserTask`" example), and
`tests/tabarena/benchmark/task/test_user_task.py`.

```python
from sklearn.model_selection import RepeatedStratifiedKFold
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import GroupLabelTypes
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits

n_folds, n_repeats = 3, 10
cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=0)
splits = from_sklearn_splits_to_user_task_splits(cv.split(X, y), n_splits=n_folds)

task = UserTask(task_name="ramanbench/bacteria-id", task_cache_path=task_cache_dir)
wrapper = task.create_task(
    dataset=df, target_feature="species", problem_type="classification", splits=splits,
    eval_metric="log_loss", group_on="specimen_id", group_labels=GroupLabelTypes.PER_GROUP,
)
task.save_task(wrapper)
task.load().validate_metadata()   # recomputes from disk and raises on any diverging field
```

The data contract `create_task` enforces:

- `dataset` is one DataFrame that includes the target column, has a default `RangeIndex`, and has
  resolved dtypes. Numeric, `category`, `string` and datetime columns are supported; `object`
  columns raise. Categorical columns must already be `category`, not strings.
- `splits` is `{repeat: {fold: (train_indices, test_indices)}}` with positional Python `int`
  lists. Numpy integers fail validation, so call `.tolist()` (the sklearn helper does). Train and
  test must not overlap, test indices must not overlap across folds of one repeat, and every repeat
  must have the same number of folds. A single holdout is `{0: {0: (train, test)}}`.
- `task_name` must be unique on the machine: the cache file, the numeric task id and the results
  `dataset` key (a slug plus a hash of the name) all derive from it.
- `eval_metric` must be an AutoGluon metric name. Defaults come from
  `tabarena.benchmark.task.metrics.DEFAULT_EVAL_METRIC_BY_PROBLEM_TYPE` (`roc_auc`, `log_loss`,
  `rmse`). To use a domain metric, register it once with `autogluon.core.metrics.make_scorer` and
  insert it into `autogluon.core.metrics.METRICS[<problem_type>]`, exactly as
  `tabarena/metrics/custom_metrics.py` does for the AMEX metric. Put the registration in a module
  every worker imports (your package `__init__` or the module that defines your tasks), otherwise
  a Ray worker resolving the metric by name will not find it.
- `group_on`, `group_labels`, `time_on`, `group_time_on`, `stratify_on`, `split_time_horizon` and
  its unit are recorded on the task. They do not build the outer splits (you supply those); they
  drive the inner validation protocol (Step 4) and the preprocessing that wants group or time
  columns (Step 7). `GroupLabelTypes.PER_GROUP` means every row of a group shares the label (one
  specimen, many spectra); `PER_SAMPLE` means the label varies within a group (one patient, many
  visits). `time_on` together with `group_on` raises `NotImplementedError` downstream, and
  `split_time_horizon` is carried and logged but no split logic reads it yet.

Where tasks live: without `task_cache_path` a `UserTask` is saved under the OpenML cache root in
`tabarena_tasks/`, which `CacheConfig(openml=...)` relocates (Step 8). A task saved there has a
path-free `task_id_str`, so `TaskMetadataCollection.from_user_tasks(tasks)` is runnable as-is. A
task saved to a custom `task_cache_path` should be handed to the runner explicitly via
`build_and_run_jobs(..., user_tasks=tasks)`, as the example does. `UserTask.from_task_id_str`
reconstructs a task from the id string, which is what lets a compute node load a task it never
created.

## Step 3: Outer splits: repeated k-fold, adaptive repeats, groups, time

TabArena does not ship its own splitter. You build the outer splits with scikit-learn (or Data
Foundry) and hand them to `create_task`. What to reproduce is the protocol:

- Repeated k-fold, not a single holdout. TabArena v0.1 uses 3 outer folds and a repeat count that
  depends on the training-set size: 10 repeats below 2,500 training rows, 1 repeat above 250,000,
  otherwise 3. The policy is `_get_n_repeats` in
  `tabarena/benchmark/task/metadata/fetch_metadata.py` (a private function; port the three lines
  rather than importing it) and the constants are restated in
  `examples/advanced/run_get_tabarena_datasets_from_openml.py`. Verify a ported policy against
  `load_curated_task_metadata()` from the same module, not just against the docstring.
- Grouped data has no repeated splitter in scikit-learn. Loop `StratifiedGroupKFold` /
  `GroupKFold` with `shuffle=True` and a different `random_state` per repeat, then feed the flat
  iterator to `from_sklearn_splits_to_user_task_splits`. Data Foundry's
  `get_recommended_grouped_splits` (in `data_foundry.curation_recommendations`) is the splitter
  TabArena itself uses for grouped inner folds, so using it for the outer splits keeps the two
  layers consistent.
- Temporal data needs forward-in-time outer splits (train strictly before test). TabArena's inner
  helper `split_time_index_into_intervals` in `tabarena/benchmark/exec_models/autogluon_utils.py`
  bins a time column into row-balanced contiguous intervals; reuse it to define the outer test
  window, then hold out the latest interval.
- Audit every dataset for group structure before the first real run. A dataset wrongly treated as
  row-independent leaks train information into test and inflates every model's score. RamanBench
  caught this late, on datasets that had already shipped without it.

Decide the repeat count from data instead of guessing. `compare(output_dir=...,
compute_fold_similarity=True, fold_similarity_kwargs={"target_reliability": 0.8})` writes
`fold_similarity.csv` with a `folds_needed_for_stability@0.8` column per dataset (a Spearman-Brown
extrapolation over the per-split rankings, from `BenchmarkEvaluator.rank_datasets_by_fold_similarity`).
Take `min(folds_needed, num_folds)` per dataset, commit the resulting `(dataset, split)` table, and
expose it as a `"core"` subset predicate with `tasks_in_frame` (Step 10). That is exactly how
BeyondArena's `core` subset was derived; the recipe is
`examples/!experimental/run_generate_beyondarena_core_subset.py`. The same mixin offers
`dataset_representativeness` and `jitter_all_datasets` for pruning redundant or noisy datasets
while curating the suite.

## Step 4: Inner validation and bagging parity

Every bagged experiment fits `num_bag_folds=8` times `num_bag_sets=1` models per outer split
(`AGModelBagExperiment` defaults, also the `default_num_folds` of
`tabarena.benchmark.task.metadata.schema.ValidationMetadata`). Datasets with at most 500 (group)
instances switch to 5 folds times 5 repeats. Match this rather than leaving bagging to whatever
AutoGluon preset resolves to: an implicit, size-dependent bagging behavior is not a reproducible
protocol and makes historical and new numbers incomparable in ways that are hard to detect later.
If an earlier version of the domain benchmark ran without explicit bagging control, say so in its
changelog rather than presenting old and new numbers as comparable.

The task's `group_on` / `time_on` only reach the inner folds when the experiment carries
`dynamic_tabarena_validation_protocol=True`. `TabArenaExperimentBundle` and
`BeyondArenaExperimentBundle` set it; `TabArenaV0pt1ExperimentBundle` turns it off for
backward compatibility. A grouped domain benchmark built on the v0.1 bundle therefore leaks
groups across bagging folds while its outer splits look correct. Use `TabArenaExperimentBundle`
(or your subclass from Step 5) for grouped or temporal data.

With the protocol on, `resolve_validation_splits` in
`tabarena/benchmark/exec_models/autogluon_utils.py` builds group-disjoint or forward-in-time inner
folds and passes them to AutoGluon as custom splits. The grouped branch imports Data Foundry, so
install `tabarena[data-foundry]` (part of `[benchmark]`). Fold counts adapt at run time: fewer
groups than folds clamps the fold count and sets one repeat, a minority class smaller than the
fold count does the same, and a `time_on` task always uses one repeat.

Two further execution modes exist for models that carry their own validation: `holdout_experiments=True`
keeps a single task-aware holdout split without bagging, and `outer_experiments=True` fits once on
all training data with no validation split (the mode most foundation models are benchmarked in;
see `examples/beyondarena/advanced/run_quickstart_beyondarena_without_bagging.py`). A method run
this way still gets the shared outer protocol and scoring, but no HPO simulation.

## Step 5: Pick the models and define your bundle

Enumerate the zoo from the registry instead of hardcoding names:

```python
from tabarena.models import get_model_registry
for key, info in get_model_registry().items():
    print(key, info.method_metadata.display_name, info.method_metadata.compute, info.pip_extra)
```

`tabarena.models.utils.get_configs_generator_from_name("TabM")` returns a model's generator, whose
`model_cls` and `manual_configs[0]` are what `examples/running_tabarena_models/run_tabarena_model.py`
uses to fit a TabArena model outside the benchmark. `tabarena.models.prefetch.prefetch_weights([...])`
downloads foundation-model checkpoints before a cluster run.

Three bundles ship; pick by protocol, not by name:

| Bundle | Random configs | Preprocessing | Task-aware validation | Time limit |
|---|---|---|---|---|
| `TabArenaV0pt1ExperimentBundle` | 200 | AutoGluon `"default"` | off | 1 h |
| `TabArenaExperimentBundle` | required field | `preprocessing_pipelines` required | on | 1 h |
| `BeyondArenaExperimentBundle` | 25 | `"tabarena_default"` | on | 4 h |

A domain benchmark should subclass `TabArenaExperimentBundle` and pin its own defaults, so every
run script shares one protocol:

```python
from dataclasses import dataclass, field
from typing import ClassVar
from tabarena.benchmark.experiment import ModelConstraints, TabArenaExperimentBundle

@dataclass(kw_only=True)
class RamanBenchExperimentBundle(TabArenaExperimentBundle):
    n_random_configs: int = 50
    preprocessing_pipelines: list[str] = field(default_factory=lambda: ["default"])
    DEFAULT_TIME_LIMIT: ClassVar[int] = 3600
    custom_model_constraints: dict[str, ModelConstraints] = field(
        default_factory=lambda: {"TA-TABICL": ModelConstraints(max_n_features=2000)},
    )
```

`ModelConstraints` (fields `max_n_features`, `max_n_samples_train_per_fold`,
`min_n_samples_train_per_fold`, `max_n_classes`, `regression_support`) drops
`(model, dataset)` jobs that violate a model's limits instead of letting them fail at fit time.
The bundle already carries TabPFNv2, TabICL and Mitra caps under their AutoGluon keys;
high-dimensional domains (spectra with thousands of wavenumbers) will hit these and should add
their own.

`models` entries are `(name_or_generator, n_configs)` with `n_configs` as `0` (default only), an
int, or `"all"`; a third tuple element pins hyperparameters into every config of that model, e.g.
`("XGBoost", 0, {"n_estimators": 100})`. `build_experiments(time_limit=..., num_cpus=...,
num_gpus=..., memory_limit=...)` bakes the compute budget into every experiment. For GPU models
pass the card's VRAM in GB as `memory_limit`: AutoGluon budgets parallel bagging folds against the
reported memory, and node RAM lets co-scheduled folds OOM the GPU (this is what `tabflow_slurm`'s
`fake_memory_for_estimates` does).

## Step 6: Model registry: reuse vs. layer your own

Do not fork TabArena's model wrappers. Two registries coexist:

TabArena's registry supplies the general-purpose zoo. A domain registry, layered in the downstream
package, holds only the models that do not belong upstream. `tabarena.models.register_model_info`
is the documented hook for this: it exists so an extension package can add `ModelInfo` entries
that `discover_models()`'s walk over `tabarena.models` cannot see. Call it from your package's
`__init__`, and the model becomes addressable by name in a bundle exactly like a built-in one.

```python
from tabarena.models import ModelInfo, register_model_info
from tabarena.models._method_metadata import ModelDescriptor
from tabarena.utils.config_utils import ConfigGenerator

spectral_cnn = ModelDescriptor(display_name="SpectralCNN", compute="gpu", is_bag=True,
                               reference_url="https://...", date_introduced="2026-03")
gen_spectral_cnn = ConfigGenerator(model_cls=SpectralCNNModel, manual_configs=[{}],
                                   search_space={...})
register_model_info(ModelInfo(
    model_cls=SpectralCNNModel,
    search_space=gen_spectral_cnn,
    method_metadata=spectral_cnn.method_metadata(
        method="SpectralCNN", ag_key="SPECCNN", config_default="SpectralCNN_c1_BAG_L1",
        suite="ramanbench-2026-09",
    ),
    pip_extra=("ramanbench[cnn]",),
))
```

Build the model class the way TabArena builds its own: an AutoGluon `AbstractModel` (or
`AbstractTorchModel` for torch) subclass with `ag_key` and `ag_name` (the generator asserts both),
`_supported_problem_types`, `_fit`, `_preprocess`, and optionally a `warmup` classmethod for
untimed one-off costs such as imports, JIT and CUDA context (the fairness contract in
`tabarena/models/warmup.py` says a warm-up may never touch task data). The `add-model` skill in
this repo describes the wrapper anatomy and points at reference models per family; it applies
unchanged to a model living in another package, minus the `pyproject.toml` edit. Config names
follow `{ag_name}_c1_BAG_L1` for the default and `_r{i}` for random configs; `config_default`
must match. Mirror `tests/tabarena/models/test_all_models.py` for a registry-driven smoke test of
your own models.

Two things bite here. A model class must live in an importable module, never in `__main__`: Ray
workers cannot unpickle it otherwise (`debug_mode=True` runs in-process and hides this). And a
model that does its own validation, tuning or ensembling (a chemometrics pipeline, an AutoML tool,
an LLM agent) is a system, not a model: subclass
`tabarena.benchmark.exec_models.ExternalSystemModel`, implement `_fit_system` / `_predict` /
`_predict_proba`, pair it with `SystemConfigGenerator`, and run the bundle with
`system_experiments=True` (`examples/advanced/run_quickstart_tabarena_external_system.py`). Its
results are recorded as baselines with no HPO simulation, which is the honest representation of a
self-tuning pipeline.

If a domain-specific model turns out to be broadly useful beyond the domain, that is a signal it
belongs upstream. Point the user at `add-model` and at Step 13.

## Step 7: Preprocessing: layer, do not fork

TabArena splits preprocessing into a model-agnostic AutoGluon feature generator (applied to the
whole frame) and a model-specific step injected into each model's hyperparameters.
`tabarena/benchmark/preprocessing/pipeline.py::resolve_preprocessing_pipeline` resolves a named
pipeline to that pair, and `build_feature_generator` forwards the task's `group_cols`,
`group_labels` and `group_time_on` to any generator class whose `__init__` accepts them.
`examples/!experimental/run_tabarena_preprocessing.py` demonstrates both stages on a frame with
numeric, categorical, text, datetime and grouped columns.

Three insertion points exist for domain preprocessing (denoising, baseline correction,
normalization, feature extraction), none of which forks a wrapper:

1. A domain feature generator. Subclass `TabArenaModelAgnosticPreprocessing` (or AutoGluon's
   `AutoMLPipelineFeatureGenerator`) and pass it as `fit_kwargs["feature_generator_cls"]` in the
   experiment's `method_kwargs`; `AGWrapper` builds it and hands it to `TabularPredictor.fit`.
   Declare `group_cols` / `group_time_on` parameters and the task's split columns arrive for free.
2. A model-specific step. `TabArenaModelSpecificPreprocessing.add_to_hyperparameters` shows how a
   generator is injected per model through `ag.model_specific_feature_generator_kwargs`; the same
   shape carries a domain step that must run after the shared generator.
3. A mixin on the model class, jointly tunable through the model's search space. This is what
   RamanBench did. Properties worth carrying over from that implementation: each tunable step gets
   its own enable flag and hyperparameters, composed through one restriction dict so a run config
   can switch steps on and off; stateful steps (anything fit on training data, e.g. a reference
   spectrum) fit once on the training fold and only transform at predict time; and a step that
   changes the feature count or column names must resync the wrapper's feature bookkeeping, since
   wrappers snapshot `X.columns` before `_fit` runs.

Named pipelines are a closed set: `Experiment._apply_preprocessing` accepts `"default"`,
`"tabarena_default"` and the experimental `FSBench__*` family and raises on anything else, so a
downstream pipeline cannot be selected by name through `preprocessing_pipelines`. Use insertion
point 1 or 2 directly on the experiments, or subclass `Experiment` and override
`_apply_preprocessing`, and note the gap for Step 13.

## Step 8: Run it: contexts, caches, distributed execution

`AbstractArenaContext` is directly instantiable. `AbstractArenaContext(task_metadata=collection,
methods=[])` is a self-contained arena with no TabArena baselines, so `compare` scores your
results alone. `build_and_run_jobs` enumerates experiments times the collection's splits (a
non-rectangular suite with different fold counts per dataset is fine), runs them, and registers the
results in memory as new methods. Scope a run with `task_subset=TaskSubset(...)` or the loose
filters `subset=`, `dataset_names=`, `split_indices="lite"`, `n_train_samples=`, and inspect the
jobs first via `build_jobs` plus `run_jobs` when you want to slice or ship them.

Caches, in `tabarena.caching.CacheConfig` (see `examples/advanced/run_configure_caches.py`):
`openml` holds materialized datasets and, under it, `tabarena_tasks/` (your `UserTask`s),
`tabarena_text_cache/` and `local/datasets/`; `huggingface` holds foundation-model weights via
`HF_HOME`; `data_foundry` holds raw container downloads; `tabarena` holds downloaded method
artifacts and leaderboards; `results` is the default `expname`. Hand one `CacheConfig` to the
context and it is applied on the driver and re-applied inside every worker. `expname` is the
runner's results directory: results land at
`{expname}/data/{method}/{task}/{repeat}_{fold}/results.pkl` (a gzipped pickle; load with
`tabarena.utils.pickle_utils.load_pickle`), and `cache_mode="default"` skips finished jobs, so
point repeated runs of one sweep at the same `expname` to resume. `expname=None` uses a throwaway
temp dir.

`debug_mode=True` runs in-process (sequential fold fitting, model classes may live in the script).
The default Ray backend needs importable model classes and enough RAM for parallel folds.
`JobBatch` serializes a sweep (`experiments.yaml`, `task_metadata.csv`, `jobs.json`,
`cache_config.json`) so a head node can ship it to compute nodes that re-run it with
`ExperimentBatchRunner.run_jobs`. On SLURM, `packages/tabflow_slurm/` already implements that
head-node flow: `TabArenaBenchmarkPlan` takes your context, your bundle, a `TaskSubset`,
`ModelJob`s with per-model resources, a `PathSetup` and a scheduler setup, resolves cache hits,
and prints the `sbatch` commands. Its `README.md` documents swapping the context for any
`AbstractArenaContext`, so a domain arena plugs in without changes. For another scheduler,
`examples/advanced/run_async_tabarena_api.py` shows the low-level fan-out API
(`context.build_jobs`, `context.run_job(job, register=False)`, `context.register(raw_results)`)
behind a swappable future-based backend.

Whatever the scheduler, keep institution-specific values (partitions, paths, venvs) in a separate
profile file (git-ignored or private) and the submission logic public and reusable, so external
contributors without cluster access can still run small jobs locally against the same tooling.

## Step 9: Evaluate: leaderboard math, imputation, Elo anchoring

`context.compare(output_dir)` writes the leaderboard CSVs, `results_per_split.csv`,
`method_info.csv`, a LaTeX table, and the figures (win-rate matrix, tuning-impact Elo bars, Pareto
fronts for Elo and improvability against train and inference time, an interactive Pareto explorer,
Elo and improvability against date introduced). `context.leaderboard(...)` computes the same table
without writing or rendering. Useful keywords: `subset=`, `datasets=`, `folds=`,
`return_results=True` (also return the per-split frame), `new_methods_only=True`,
`return_single=True` (one method's row as a Series, scored against every baseline),
`average_seeds=`, `remove_imputed=True`, `score_on_val=True`, `compute_fold_similarity=True`
(Step 3). `compare_per_dataset` and `generate_per_dataset_tables` produce per-dataset views.

Imputation is not optional. bencheval refuses non-dense input (every method must have a row for
every `(task, split)` it was evaluated on), so failed fits must be imputed before scoring.
`compare(fillna=...)` (default `"auto"`, which reads the context's `fillna_method`) fills a method's
missing splits with a named method's rows and marks them in an `imputed` column. TabArena uses its
default random forest; a self-contained arena has to name one of its own methods or leave
`fillna_method=None`, in which case any failure surfaces as a dense-data error. Pick a cheap,
reliable, weak model you always run (a default random forest is the usual choice) and keep the
imputed fraction visible in the leaderboard.

Elo is pairwise and pool-relative. `calibration_method` (context) / `calibration_framework` (Elo
kwargs) anchors one method at 1000 so numbers stay readable across runs; with `methods=[]` set it
to one of your methods or to `None`. Bootstrap rounds default to 100 in bencheval and 200 in
TabArena's reporter; keep them fixed across published leaderboards and record them.

When the domain benchmark computes its own results (no TabArena runner), call bencheval directly
(`examples/plots/run_generate_custom_leaderboard.py` is the complete tutorial):

```python
from bencheval.evaluator import BenchmarkEvaluator, LeaderboardContext, LeaderboardMetric

ev = BenchmarkEvaluator(task_col="dataset", seed_column="fold",
                        columns_to_agg_extra=["time_train_s", "time_infer_s"])
df = ev.fillna_data(df, fillna_method="RandomForest (default)", imputed_col="imputed")
lb = ev.leaderboard(df, include_error=True,
                    elo_kwargs=dict(calibration_framework="RandomForest (default)", calibration_elo=1000),
                    metrics=["elo", "winrate", "improvability", LeaderboardMetric("worst_task", my_metric)])
```

Contracts: one row per `(method, task[, seed])`; `metric_error` numeric, non-negative, lower is
better (convert a higher-is-better score yourself, e.g. `1 - score`); `columns_to_agg_extra="auto"`
silently requires `time_train_s` and `time_infer_s`, so pass `None` when you have no timings;
`leaderboard()` validates but never imputes. A `LeaderboardMetric` receives a `LeaderboardContext`
with the per-task results and returns method-indexed Series, so a domain metric becomes a
leaderboard column without forking bencheval. The per-1k-rows time columns TabArena's website shows
are computed by the caller before scoring (`time_train_s * 1000 / n_train_rows`, then listed in
`columns_to_agg_extra`); see `tabarena/evaluation/leaderboard_reporter.py` for the exact wiring.

## Step 10: Your own arena: context, dataset catalogue, method roster, hosting

For a domain that wants the full native experience (named presets, subset predicates, hosted
baselines everyone compares against, website artifacts), subclass `AbstractArenaContext` the way
`BeyondArenaContext` does. The module docstring of `tabarena/contexts/beyondarena/context.py`
lists the only three things that differ between arenas: subset predicates, task metadata, method
metadata. Everything else is inherited.

```python
import pandas as pd
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.benchmark.task.subset_predicate import SubsetPredicate, tasks_in_frame
from tabarena.contexts import AbstractArenaContext

class RamanBenchContext(AbstractArenaContext):
    benchmark_name = "RamanBench"
    SUBSET_PREDICATES = {
        **AbstractArenaContext.SUBSET_PREDICATES,
        "high-dim": SubsetPredicate(lambda df: df["n_features"] > 1000, ("n_features",)),
        "grouped": SubsetPredicate(lambda df: df["task_type"] == "grouped", ("task_type",)),
        "core": tasks_in_frame(pd.read_csv(CORE_TASKS_CSV)),
    }
    SUBSET_SHORTCUTS = {"full": [], "core": ["core"]}

    def __init__(self, methods="ramanbench", task_metadata="ramanbench", *,
                 fillna_method="RandomForest (default)", calibration_method="RandomForest (default)",
                 **kwargs):
        super().__init__(methods=methods, task_metadata=task_metadata,
                         fillna_method=fillna_method, calibration_method=calibration_method, **kwargs)

    def _resolve_task_metadata_preset(self, name):
        return TaskMetadataCollection.from_source(RamanBenchTaskMetadataSource())

    def _resolve_methods_preset(self, name):
        return list(ramanbench_method_metadata_lst)
```

Predicates evaluate against `TaskMetadataCollection.task_grid()`, whose columns are `dataset`,
`fold`, `repeat`, `split`, `max_train_rows`, `n_features`, `n_classes`, `problem_type`,
`task_type`, `num_cols_after_preprocessing`, `num_text_cols`, `num_high_cardinality_cats`,
`has_categorical`, `has_datetime`, `group_labels`. A `SubsetPredicate` declares the columns it
reads so a missing column fails with a clear message. `tasks_in_frame` turns a committed
`(dataset, split)` CSV into a predicate, which is how data-dependent subsets like `core` work.

The dataset catalogue is a `TaskMetadataSource` (`tabarena/benchmark/task/metadata/sources/base.py`):
`load()` returns the suite's `TabArenaTaskMetadata` cheaply from a committed CSV so users can filter
before downloading anything, and `materialize(tasks)` downloads and converts only the surviving
tasks into `UserTask`s. `DataFoundryTaskMetadataSource` is the reference implementation, and
`scripts/generate_beyond_arena_metadata.py` shows how the committed CSV is regenerated
(`TaskMetadataCollection.to_dataframe()` round-trips through `from_source`). The suite registry
that maps a string like `"BeyondArena"` to a source is private, so a domain source is passed as an
instance, as above.

The method roster is a `methods.py` like `tabarena/contexts/beyondarena/methods.py`: about 150
lines that re-key the whole zoo into your suite by calling each model's `ModelDescriptor`
(`from tabarena.models.<key>.info import <key>_descriptor`) with
`.method_metadata(method=..., ag_key=..., config_default=..., suite="ramanbench-2026-09",
cache_type=..., cache_kwargs=...)`. Intrinsic facts (display name, paper, compute class, bagging)
stay upstream; only run-specific fields are yours. `can_hpo` is inferred from how many configs you
ran, `date` records the run, `verified=False` marks results not yet signed off, and
`method_class="system"` plus `tags` (`"with-llm"`, `"closed-source-api"`) place systems in the
right entrant pools (`tabarena/evaluation/entrants.py` explains why entrant pools are separate
evaluations rather than a display filter).

Hosting: `MethodMetadata.cache_type` is `"local"`, `"s3"` or `"r2"` with the bucket and prefix in
`cache_kwargs`; `artifact_dir` points at a committed copy in the downstream repo's `data/` folder,
bypassing the cache layout entirely and suitable for small suites. Build the artifacts from a run
directory with `EndToEnd.from_path_raw(path_raw, cache_raw=True, cache_processed=True)`
(`examples/plots/run_end_to_end_from_raw.py`), then `method_uploader()` pushes them. Three tiers
exist: raw (per-fold predictions, roughly 100 GB per method on TabArena), processed (an
`EvaluationRepository` for HPO and ensemble simulation), and results (small DataFrames). Hosting
only the results tier already lets others run `YourContext(extra_methods=...).compare(...)`
against your baselines.

Website: `context.leaderboard_to_website_format(leaderboard)` produces the published column set via
`tabarena/website/website_format.py::format_leaderboard`. `get_model_family` there classifies
methods by name prefix into Tree-based, Foundation Model, Neural Network, Baseline, System; a
domain model with an unknown prefix lands in "Other" until the table is extended upstream
(Step 13). The self-contained HTML explorers in `tabarena/plot/interactive/` (`build_pareto_explorer_html`,
`build_leaderboard_explorer_html`, `build_winrate_explorer_html`, `build_per_dataset_explorer_html`)
take plain DataFrames, as do `tabarena/plot/subset_results.py::plot_subset_results` and
`tabarena/plot/composite_leaderboard.py::generate_composite_leaderboard` for a per-subset overview.
`scripts/run_generate_beyondarena_website_artifacts.py` is the compact reference for a
subset-axis leaderboard site.

## Step 11: Packaging: PyPI release or git URL, deliberately

Both `tabarena` and `bencheval` publish PyPI pre-releases, so a downstream package has two options:

- Pin the PyPI release (`tabarena>=0.1.0a1`, `bencheval>=0.1.0a1`; check `pip index versions
  --pre tabarena`) if the domain benchmark can trail the release cadence. This is PyPI-clean and
  installs are ordinary. TabArena depends on an AutoGluon pre-release, so users need `--pre`, or an
  exact `autogluon.tabular==<version>` pin, which resolves a pre-release without the flag. Model
  extras whose upstream is git-only (`tabfm`, `sap-rpt-oss`, `exaone_tabular`) are empty on PyPI;
  the model's `pip_extra` in its `info.py` keeps the exact git dependency for the install hint.
- Track `main` via a git URL (`tabarena @ git+https://github.com/autogluon/tabarena.git#subdirectory=packages/tabarena`)
  if you need a fix or hook that is not released yet. PyPI rejects any distribution that declares a
  direct URL dependency, in any extra, so a package pinning a git URL anywhere cannot itself publish
  to PyPI (`Can't have direct dependency`). Isolate it into its own optional extra (e.g.
  `[benchmark]`), keep the PyPI-clean extras separate, and document that full-functionality installs
  stay from-source (`pip install -e .[full]`). `scripts/release/build_pypi_dists.py` shows how
  TabArena itself strips direct URLs from a staged copy before building; copy that if you want a
  PyPI build that stays in sync with a git-tracking source tree.

## Step 12: Reproducibility and hardware

Pin the method roster you publish against, the way
`examples/reproducibility/run_generate_main_leaderboard_neurips2025.py` passes a frozen
collection (`TabArenaContext(methods=<dated collection>.method_metadata_lst)`), and keep dated
roster modules as the suite grows. Pin `tabarena` and `bencheval` at one version per published
leaderboard, record the Elo bootstrap rounds, and note that
the `TABARENA_LEGACY_ELO_SOLVER_TOL=1` environment variable reproduces Elo numbers published
before bencheval tightened its solver tolerance (a temporary flag, per its docstring).

Time columns are not hardware-normalized anywhere in the stack. Every `results.pkl` carries
`experiment_metadata` (timings, warm-up time) and `method_metadata` (`num_cpus`, `num_gpus`,
per-child resources), but nothing records the hardware itself. Record a hardware descriptor per run
in your own results from day one (machine label, CPU model, GPU model, driver), since it cannot be
recovered later. `scripts/hardware_normalization_design.md` in this repo documents the agreed but
unimplemented normalization design and the measurement hygiene that turned out to matter (exclusive
nodes, discard the first repetition, pin iteration counts, size-dependent factors);
`scripts/hardware_probe.py` is a standalone, numpy-only machine profiler you can copy.

For unit tests of your own analysis code, `tabarena.simulation.context_artificial.load_repo_artificial()`
builds a synthetic `EvaluationRepository`, and `tests/bencheval/test_evaluator.py` holds a
hand-computable 3x3 fixture with expected values for every leaderboard column.

## Step 13: Tell the TabArena maintainers what you needed

Do this early, ideally after Step 0, and again when the benchmark is public. TabArena's API is
still moving, and the maintainers add hooks when a downstream benchmark shows where the current
surface forces a workaround. Open an issue at `https://github.com/autogluon/tabarena/issues`
(or a discussion) titled after your benchmark, and include:

- what the domain is, where the repo lives, and which packages and version you depend on;
- the list of private or undocumented things you had to import (`tabarena.models._method_metadata`
  for `ModelDescriptor` and the enums, `tabarena.models._method_metadata_collection`, the private
  `_get_n_repeats`, the suite registry in `tabarena.benchmark.task.metadata.sources`);
- every place you subclassed or monkey-patched to get past a closed set, with the one-line reason;
- protocol features your domain needs that the runner does not support yet;
- the models or systems you registered locally that could belong upstream (then use `add-model` /
  `add-system` to contribute them, which also gets them benchmarked on TabArena and BeyondArena).

Gaps this survey found that a domain benchmark is likely to hit, phrased as requests:

- a registry for named preprocessing pipelines (today `Experiment._apply_preprocessing` accepts
  only `"default"`, `"tabarena_default"` and `FSBench__*`);
- a public way to register a `TaskMetadataSource` under a suite name, so
  `TaskMetadataCollection.from_preset("RamanBench")` works like `"BeyondArena"`;
- a hook for model families in `website_format.get_model_family`, so a domain model is not shown
  as "Other";
- `time_on` together with `group_on` in the validation protocol, and honoring `split_time_horizon`
  in the inner temporal split;
- grouped inner splits without the `data-foundry` dependency;
- public homes for `ModelDescriptor`, `MethodMetadataCollection` and `InMemoryMethodMetadata`,
  and a public `get_n_repeats`;
- a documented hardware descriptor in `experiment_metadata` (see Step 12).

Mention that you are willing to test a prerelease branch. Maintainers can also link the domain
benchmark from TabArena's examples README, and a benchmark that keeps protocol parity is easy to
cross-reference with TabArena's own leaderboard.

## Step 14: Report

Summarize for the user:

- which package(s) the downstream benchmark depends on (`tabarena`, `bencheval`, or both), which
  extras, and whether via PyPI release or git URL (Step 11);
- how datasets become `UserTask`s, which datasets carry `group_on` / `time_on`, and which needed
  the group-structure audit;
- the outer split protocol (folds, repeat policy, grouped or temporal) and whether repeats were
  derived from fold-stability analysis;
- the bundle subclass and its defaults (configs, time limit, task-aware validation on, model
  constraints), and any historical-vs-new comparability caveat for the domain benchmark's changelog;
- where the domain model registry lives relative to TabArena's own and which models were
  registered or wrapped as systems;
- how domain preprocessing is layered (feature generator, model-specific step, or mixin);
- the imputation method and Elo anchor the leaderboard uses;
- whether a domain arena context, catalogue source and method roster exist, and where results are
  hosted;
- the list of API gaps and workarounds to report upstream (Step 13).

## Gotchas collected along the way

- `object` dtype columns raise in `create_task`; convert to `string` or `category` first.
- Split indices must be Python `int`s; numpy integers fail validation.
- `TabArenaV0pt1ExperimentBundle` disables the task-aware validation protocol; grouped data on it
  leaks groups across bagging folds.
- Grouped inner splits import Data Foundry; without `tabarena[data-foundry]` a `group_on` task
  fails at fit time.
- `subset="core"` and the other data-dependent predicates are specific to the official suites;
  size, problem-type and split-regime predicates work on any collection.
- With `methods=[]`, `fillna_method` and `calibration_method` must be `None` or one of your own
  methods; the TabArena defaults name baselines you do not have.
- bencheval never imputes; `columns_to_agg_extra="auto"` requires the two timing columns; passing
  `metrics=` disables every `include_*` flag.
- bencheval defaults to `task` / `seed` column names while TabArena frames use `dataset` / `fold`;
  set `task_col` and `seed_column` accordingly.
- `results.pkl` files are gzipped pickles; `pickle.load` on the raw file fails with an invalid load
  key.
- A model class defined in `__main__` only works with `debug_mode=True`.
- The `plot` extra pins `matplotlib` and `seaborn` exactly and needs `autorank` for critical
  difference diagrams; keep those pins out of your core install.
