
<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <img src="https://avatars.githubusercontent.com/u/210855230" width="175" alt="TabArena Logo"/>
    </summary>
  </ul>
</div>

## A Living Benchmark for Machine Learning on Tabular Data 💫

---

| 🚀 [Leaderboard](https://tabarena.ai/) | 📂 [Example Scripts]( https://tabarena.ai/code-examples) | 📊 [Dataset Curation](https://tabarena.ai/data-tabular-ml-iid-study) | 📄 [Paper](https://tabarena.ai/paper-tabular-ml-iid-study) |
|:--------------------------------------:|:----------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|

---
</div>

TabArena is a living benchmarking system that makes benchmarking tabular machine learning models a reliable experience. TabArena implements best practices to ensure methods are represented at their peak potential, including cross-validated ensembles, strong hyperparameter search spaces contributed by the method authors, early stopping, model refitting, parallel bagging, memory usage estimation, and more.

TabArena currently consists of:

- 51 manually curated tabular datasets representing real-world tabular data tasks.
- 9 to 30 evaluated splits per dataset.
- 27+ tabular machine learning methods, including 10+ tabular foundation models.
- More than 50 million trained models across the benchmark, with all validation and test predictions cached to enable tuning and post-hoc ensembling analysis.
- A [live TabArena leaderboard](https://huggingface.co/spaces/TabArena/leaderboard) showcasing the results.


## ⚡ Quickstart

> [!TIP]
> The fastest way to try TabArena end-to-end:

```bash
pip install uv
git clone https://github.com/autogluon/tabarena.git && cd tabarena
uv sync --extra benchmark
uv run python examples/benchmarking/run_quickstart_tabarena.py
```

For other install paths (eval-only, editable AutoGluon, dependency), see [Installation](#-installation) below.

## 🕹️ Use Cases

We share more details on various use cases of TabArena in our [examples](examples):

* 📊 **Benchmarking Predictive Machine Learning Models**: please refer to [examples/benchmarking](examples/benchmarking).
* 🚀 **Using SOTA Tabular Models Benchmarked by TabArena**: please refer to [examples/running_tabarena_models](examples/running_tabarena_models).
* 🗃️ **Analysing Metadata and Meta-Learning**: please refer to [examples/meta](examples/meta).
* 📈 **Generating Plots and Leaderboards**: please refer to [examples/plots_and_leaderboards](examples/plots_and_leaderboards).
* 🔁 **Reproducibility**: we share instructions for reproducibility in [examples](examples).

### Datasets

Please refer to our [dataset curation repository](https://github.com/TabArena/tabarena_dataset_curation) to learn more about or contributed data!

### More Documentation

TabArena code is currently being polished. Detailed Documentation for TabArena will be available soon.

# 🪄 Installation

> [!IMPORTANT]
> Requires Python **3.11–3.13** and [uv](https://docs.astral.sh/uv/getting-started/installation/).

Pick the install path that matches what you want to do:

<details>
<summary><b>📊 Evaluation only</b> — leaderboards & metrics, no model fitting</summary>

```bash
git clone https://github.com/autogluon/tabarena.git
cd tabarena
uv sync
```
</details>

<details>
<summary><b>🚀 Benchmark</b> — core set of models for benchmarking</summary>

Installs the core models used for standard benchmarking: `tabpfn`, `tabicl`, `ebm`, `search_spaces`, `realmlp`, `tabdpt`, `tabm`.

```bash
git clone https://github.com/autogluon/tabarena.git
cd tabarena
uv sync --extra benchmark
```
</details>

<details>
<summary><b>➕ Benchmark + Extended</b> — core models plus the extended model set</summary>

> The `extended` extra is **experimental** and may fail to resolve or install due to incompatible version requirements 
> across model dependencies. Use it only if you specifically need every model in a single environment; 
> otherwise prefer `benchmark` or `benchmark` plus one specific model.

Layers the extended model set (`modernnca`, `xrfm`, `sap-rpt-oss`, ...) on top of the core benchmark set.

```bash
git clone https://github.com/autogluon/tabarena.git
cd tabarena
uv sync --extra benchmark --extra extended
```

To install only one extended model on top of `benchmark` (recommended over `extended` when you only need a single extra model), pass its extra by name — for example, just `xrfm`:

```bash
git clone https://github.com/autogluon/tabarena.git
cd tabarena
uv sync --extra benchmark --extra xrfm
```
</details>

<details>
<summary><b>🛠️ Developer</b> — editable AutoGluon + editable TabArena</summary>

Create a virtual environment:

```bash
uv venv --seed --python 3.12 ~/.venvs/tabarena
source ~/.venvs/tabarena/bin/activate
```

Install editable AutoGluon and TabArena:

```bash
git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh

git clone https://github.com/autogluon/tabarena.git
uv pip install --prerelease=allow -e "./tabarena/tabarena[benchmark]"
```

> In PyCharm, mark `tabarena/` and each `autogluon/src/` subdirectory as **Sources Root** so imports resolve.

</details>

<details>
<summary><b>📦 Use TabArena as a dependency</b></summary>

Add the following to your project's dependencies:

```toml
"tabarena @ git+https://github.com/autogluon/tabarena.git#subdirectory=tabarena"
```

# 📦 TabArena Artifacts

TabArena caches predictions, results, and leaderboards as downloadable artifacts so you can reproduce or extend any analysis without re-running the benchmark.

<details>
<summary><b>Artifact tiers, sizes, and examples</b></summary>

> Artifacts download to `~/.cache/tabarena/` by default. Override the location with the `TABARENA_CACHE` environment variable.
> 
> Raw data is **~100 GB per method type**. Point `TABARENA_CACHE` at a large disk before downloading it.

| Tier | Contents | Size / method | Example |
|---|---|---|---|
| **Raw data** | Per-child test predictions, full metadata, system info | ~100 GB | [`inspect_raw_data.py`](examples/meta/inspect_raw_data.py) |
| **Processed data** | Minimal data for HPO simulation, portfolios, leaderboards | ~10 GB | [`inspect_processed_data.py`](examples/meta/inspect_processed_data.py) |
| **Results** | Per-config / HPO DataFrames (test error, val error, train time, inference time) | <1 MB | [`run_generate_main_leaderboard.py`](examples/plots/run_generate_main_leaderboard.py) |
| **Leaderboards** | Aggregated ELO, win-rate, average rank, improvability | <1 MB | — |
| **Figures & Plots** | Generated from results and leaderboards | — | — |

</details>


# 📄 Citation

> [!TIP]
> If you use TabArena in a scientific publication, please cite our paper.

**TabArena: A Living Benchmark for Machine Learning on Tabular Data**
Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas, Frank Hutter
*NeurIPS 2025, Datasets and Benchmarks Track*

📄 [arXiv](https://arxiv.org/abs/2506.16791) · 🎤 [NeurIPS poster & video](https://neurips.cc/virtual/2025/loc/san-diego/poster/121499)

<details>
<summary><b>BibTeX</b></summary>

> The entry uses `year=2026` because NeurIPS'25 proceedings are published in 2026.

```bibtex
@article{erickson2026tabarena,
  title   = {TabArena: A Living Benchmark for Machine Learning on Tabular Data},
  author  = {Erickson, Nick and Purucker, Lennart and Tschalzev, Andrej and Holzm{\"u}ller, David and Desai, Prateek and Salinas, David and Hutter, Frank},
  journal = {Advances in Neural Information Processing Systems},
  volume  = {38},
  year    = {2026}
}
```

</details>


--- 
## Relation to TabRepo 

TabArena was built upon and now replaces [TabRepo](https://arxiv.org/pdf/2311.02971). To see details about TabRepo, the portfolio simulation repository, refer to [tabrepo.md](tabrepo.md).
