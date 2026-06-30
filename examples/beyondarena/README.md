<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <img src="https://avatars.githubusercontent.com/u/210855230" width="125" alt="TabArena Logo"/>
    </summary>
  </ul>
</div>

## BeyondArena Examples 🌍

</div>

**BeyondArena** is the second benchmark that ships in this repository, built on the same code as
[TabArena](https://tabarena.ai). It is the first unified, holistic benchmark for tabular data that
goes **beyond the IID assumption**: it spans diverse task types (**IID / temporal / grouped**
splits) across a wide range of dataset **sizes** and **feature dimensionalities**, so you can see
how methods hold up on the kind of non-IID, large, and high-dimensional data that real-world
tabular problems actually look like.

It comes from the paper
[*Beyond IID: How General Are Tabular Foundation Models, Really?*](https://arxiv.org/abs/2606.30410),
which evaluates 11 models across 142 datasets and finds that today's tabular foundation models excel
on tiny-to-medium IID data, while traditional tree-based and deep-learning models still dominate on
non-IID, large, and high-dimensional datasets.

> If you already know TabArena, you know BeyondArena: it uses the **same** experiment/runner/eval
> API. The only swap is the context — `BeyondArenaContext` instead of `TabArenaContext`, and
> `BeyondArenaExperimentBundle` instead of `TabArenaV0pt1ExperimentBundle`. The datasets are
> downloaded and curated on demand via [Data Foundry](#about-data-foundry).

### ⚡ Quickstart

Requires the `benchmark` install (which includes the `data-foundry` extra used to fetch datasets):

```bash
uv pip install --prerelease=allow -e "./packages/tabarena[benchmark]"
python examples/beyondarena/run_quickstart_beyondarena.py
```

This benchmarks a custom model on a small, fast slice of BeyondArena and prints a leaderboard
comparing it against the cached BeyondArena baselines. Like every example here, it evaluates on the
recommended **`core`** subset (further narrowed to `tiny`, `!high-dim` just to keep the quickstart
fast).

## 🕹️ Use Cases

### 🚀 Benchmark your model on BeyondArena

- `run_quickstart_beyondarena.py` — Run a custom (non-registry) model through the full BeyondArena
  loop (TabArena preprocessing + the split-regime-aware validation protocol, with bagging) and
  compare it against the cached BeyondArena baselines.
- `run_generate_beyondarena_leaderboard.py` — Generate the official BeyondArena leaderboard from the
  cached results, in the website format.

### 🧪 Advanced and Specialized Usage

The `advanced/` folder holds lower-level and specialized workflows:

- `advanced/run_quickstart_beyondarena_without_bagging.py` — The same loop as the quickstart, but as
  *outer* (no train/val split, no bagging) experiments via `outer_experiments=True`.
- `advanced/run_quickstart_beyondarena_custom_datasets.py` — Benchmark on your **own** dataset:
  load a [Data Foundry](#about-data-foundry) container, convert it to a TabArena `UserTask`, and
  evaluate models on it (no BeyondArena baselines, leaderboard computed purely from your results).

## 🔎 Choosing what to run: subsets

> **Default to `core` — it's our recommended subset, and what all examples use.** `core` already
> yields stable rankings, so you do **not** need to run the full `all` split set. Layer other
> predicates on top to focus a run (e.g. `["core", "regression"]`, `["core", "tiny"]`). The one
> exception is your *own* custom datasets, where `core` does not apply — see
> [`advanced/run_quickstart_beyondarena_custom_datasets.py`](advanced/run_quickstart_beyondarena_custom_datasets.py).

`BeyondArenaContext` scopes a run with the `subset=` argument. Pass a single predicate name or a
list; combine them within an expression with `|` (OR) and `!` (NOT), and across a list with AND
(e.g. `subset=["core", "tiny", "!high-dim"]` = core **and** tiny **and** not high-dim). The full set
of predicates (see `BeyondArenaContext.SUBSET_PREDICATES`):

| Group | Predicates | Meaning |
|---|---|---|
| **Problem type** | `binary`, `multiclass`, `classification`, `regression` | The task's target type. |
| **Size bucket** | `tiny`, `small`, `medium`, `large` | By `max_train_rows`: tiny ≤ 1k, small ≤ 10k, medium ≤ 100k, large ≤ 1M. |
| **Split regime** | `iid` (alias `random`), `temporal`, `grouped` | How the splits are drawn — the *beyond-IID* axis. |
| **Features** | `low-dim`, `high-dim`, `text`, `high-cardinality` | `low/high-dim` split at 100 cols after preprocessing; `text`/`high-cardinality` keep datasets that have such columns. |
| **Split** | `core`, `lite`, `all` | `core` = each dataset's first `folds_to_use` splits (**the recommended default — use this**); `lite` = the first split only (fast smoke test); `all` = every split (rarely needed — `core` is enough). |

`core` is data-dependent: it keeps each dataset's first `folds_to_use` splits, where
`folds_to_use = min(folds_needed_for_stability, num_folds)` from a fold-similarity analysis.
It is the recommended default because it uses only as many splits as are needed for stable rankings — running the full `all` set costs much more
compute for no meaningful change in the leaderboard.

<a name="about-data-foundry"></a>
## 📦 About Data Foundry

BeyondArena's datasets are curated and distributed through **Data Foundry**, a framework for curating
tabular datasets introduced alongside the benchmark. The `BeyondArenaContext` ships only small
committed *metadata* (so you can filter by size / problem type / split regime *before* downloading
anything); when you run jobs, the selected datasets are downloaded from HuggingFace and converted on
demand. Data Foundry is pulled in by the `data-foundry` extra, which is part of the `benchmark`
install. The custom-datasets example shows how to bring your own Data Foundry container.

## 📄 Citation

If you use BeyondArena in a scientific publication, please cite:

**Beyond IID: How General Are Tabular Foundation Models, Really?**
Lennart Purucker, Andrej Tschalzev, Nick Erickson, Gioia Blayer, David Holzmüller, Alan Arazi,
Alexander Pfefferle, Mustafa Tajjar, Gaël Varoquaux, Frank Hutter

📄 [arXiv](https://arxiv.org/abs/2606.30410)

```bibtex
@misc{purucker2026beyondiid,
  title         = {Beyond IID: How General Are Tabular Foundation Models, Really?},
  author        = {Purucker, Lennart and Tschalzev, Andrej and Erickson, Nick and Blayer, Gioia and Holzm{\"u}ller, David and Arazi, Alan and Pfefferle, Alexander and Tajjar, Mustafa and Varoquaux, Ga{\"e}l and Hutter, Frank},
  year          = {2026},
  eprint        = {2606.30410},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2606.30410}
}
```
