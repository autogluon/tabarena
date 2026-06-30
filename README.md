
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

| 🚀 [Leaderboard](https://tabarena.ai/) | 📂 [Example Scripts]( https://tabarena.ai/code-examples) | 📊 [Dataset Curation](https://tabarena.github.io/data-foundry/) | 📄 Papers: [TabArena](https://arxiv.org/abs/2506.16791) · [BeyondArena](https://arxiv.org/abs/2606.30410) |
|:--------------------------------------:|:----------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|

---
</div>

TabArena is a living benchmarking system that makes benchmarking tabular machine learning models a reliable experience. TabArena implements best practices to ensure methods are represented at their peak potential, including cross-validated ensembles, strong hyperparameter search spaces contributed by the method authors, early stopping, model refitting, parallel bagging, memory usage estimation, and more. Explore the latest results on the [live leaderboard](https://huggingface.co/spaces/TabArena/leaderboard).

This single codebase powers **two complementary benchmarks** that share the same fitting, runner, and
evaluation code:

- 🏟️ **TabArena-v0.1** — the living benchmark on **curated, IID** tabular datasets.
- 🌍 **BeyondArena** — a holistic, ***beyond-IID*** benchmark spanning **IID, temporal, and grouped**
  tasks across a wide range of dataset sizes and feature dimensionalities. 
  **BeyondArena will superseed TabArena-v0.1 in the future.**

> [!TIP]
> **New here? Start with TabArena, then graduate to BeyondArena.** Get your model working and
> competitive on TabArena's curated IID datasets first; once it holds up there, run the *same* code
> on BeyondArena to stress-test how well it generalizes beyond IID.

**TabArena** covers 51 curated datasets (9–30 splits each) and 27+ methods, including 10+ tabular
foundation models — over 50M trained models, with all validation and test predictions cached for
tuning and post-hoc ensembling. **BeyondArena** extends this to **[142 datasets](https://huggingface.co/datasets/TabArena/BeyondArena)** across IID,
temporal, and grouped task types, spanning tiny to 1M-row datasets and low- to high-dimensional
features.


## ⚡ Quickstart

> [!TIP]
> The fastest way to try TabArena end-to-end:

```bash
pip install uv
git clone https://github.com/autogluon/tabarena.git && cd tabarena
uv venv --seed --python 3.12 && source .venv/bin/activate
uv pip install --prerelease=allow -e "./packages/tabarena[benchmark]"
python examples/benchmarking/run_quickstart_tabarena.py
```

For other install paths (eval-only, editable AutoGluon, dependency), see [Installation](#-installation) below.
To try **BeyondArena** instead, run `python examples/beyondarena/run_quickstart_beyondarena.py` with the same install.

## 🕹️ Use Cases

We share more details on various use cases of TabArena in our [examples](examples):

* 🌍 **Benchmarking Beyond IID (BeyondArena)**: please refer to [examples/beyondarena](examples/beyondarena).
* 📊 **Benchmarking Predictive Machine Learning Models**: please refer to [examples/benchmarking](examples/benchmarking).
* 🚀 **Using SOTA Tabular Models Benchmarked by TabArena**: please refer to [examples/running_tabarena_models](examples/running_tabarena_models).
* 🧪 **Advanced and Specialized Usage**: please refer to [examples/advanced](examples/advanced).
* 🗃️ **Analysing Metadata and Meta-Learning**: please refer to [examples/meta](examples/meta).
* 📈 **Generating Plots and Leaderboards**: please refer to [examples/plots](examples/plots).
* 🔁 **Reproducibility**: we share instructions for reproducibility in [examples](examples).

### Datasets

Please refer to our [dataset curation repository](https://github.com/TabArena/tabarena_dataset_curation) to learn more about or contributed data!

### More Documentation

TabArena code is currently being polished. Detailed Documentation for TabArena will be available soon.

# 🪄 Installation

> [!IMPORTANT]
> Requires Python **3.11–3.13** and [uv](https://docs.astral.sh/uv/getting-started/installation/).

TabArena is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/); its installable
packages live under `packages/` (`tabarena`, `bencheval`, `tabflow_slurm`). Install the `tabarena`
package directly from `packages/tabarena` with the extras you need. The `--prerelease=allow` flag is
required so uv resolves the pre-release dependency.

First clone the repo and create a virtual environment (one time):

```bash
git clone https://github.com/autogluon/tabarena.git
cd tabarena
uv venv --seed --python 3.12
source .venv/bin/activate
```

Then pick the install path that matches what you want to do:

<details>
<summary><b>📊 Evaluation only</b> — leaderboards, metrics & plots, no model fitting</summary>

Loads cached results and computes/plots leaderboards & metrics (ELO, win-rates, ranks). Depends on `autogluon.tabular` (not the full AutoGluon meta-package) — no model-fitting libraries and no torch.

```bash
uv pip install --prerelease=allow -e "./packages/tabarena[plot]"
```
</details>

<details>
<summary><b>🚀 Benchmark</b> — core set of models for benchmarking</summary>

Installs the core models used for standard benchmarking: `tabpfn`, `tabicl`, `ebm`, `search_spaces`, `realmlp`, `tabdpt`, `tabm`.

```bash
uv pip install --prerelease=allow -e "./packages/tabarena[benchmark]"
```
</details>

<details>
<summary><b>➕ Benchmark + Extended</b> — core models plus the extended model set</summary>

> The `extended` extra is **experimental** and may fail to resolve or install due to incompatible version requirements 
> across model dependencies. Use it only if you specifically need every model in a single environment; 
> otherwise prefer `benchmark` or `benchmark` plus one specific model.

Layers the extended model set (`modernnca`, `xrfm`, `sap-rpt-oss`, ...) on top of the core benchmark set.

```bash
uv pip install --prerelease=allow -e "./packages/tabarena[benchmark,extended]"
```

To install only one extended model on top of `benchmark` (recommended over `extended` when you only need a single extra model), pass its extra by name — for example, just `xrfm`:

```bash
uv pip install --prerelease=allow -e "./packages/tabarena[benchmark,xrfm]"
```
</details>

<details>
<summary><b>🛠️ Developer</b> — editable AutoGluon + editable TabArena</summary>

Create a virtual environment in your workspace directory (it spans both repos cloned below, so `.venv` lives at the workspace root rather than inside either repo):

```bash
uv venv --seed --python 3.12 .venv
source .venv/bin/activate
```

Install editable AutoGluon and TabArena:

```bash
git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh

git clone https://github.com/autogluon/tabarena.git
uv pip install --prerelease=allow -e "./tabarena/packages/tabarena[benchmark]"
```

> In PyCharm, mark `packages/tabarena/src/` and each `autogluon/src/` subdirectory as **Sources Root** so imports resolve.

</details>

<details>
<summary><b>📦 Use TabArena as a dependency</b></summary>

Add the following to your project's dependencies:

```toml
# TabArena depends on a pre-release of AutoGluon, so allow pre-releases when installing
# (e.g. `uv pip install --prerelease=allow ...` or `pip install --pre ...`).
# Alternatively, pin AutoGluon to a specific pre-release (an exact `==` pin resolves a
# pre-release without the flag), e.g. add `"autogluon.tabular==1.5.1b20260626"`.
"tabarena @ git+https://github.com/autogluon/tabarena.git#subdirectory=packages/tabarena"
```

</details>

# 📦 TabArena Artifacts

TabArena caches predictions, results, and leaderboards as downloadable artifacts so you can reproduce or extend any analysis without re-running the benchmark.

<details>
<summary><b>Artifact tiers, sizes, and examples</b></summary>

> Artifacts download to `~/.cache/tabarena/` by default. Override the location with the `TABARENA_CACHE` environment variable.
> 
> Raw data is **~100 GB per method type**. Point `TABARENA_CACHE` at a large disk before downloading it.

| Tier | Contents | Size / method | Example |
|---|---|---|---|
| **Raw data** | Per-child test predictions, full metadata, system info | ~100 GB | [`inspect_raw_data_and_verify_splits.py`](examples/meta/inspect_raw_data_and_verify_splits.py) |
| **Processed data** | Minimal data for HPO simulation, portfolios, leaderboards | ~10 GB | [`inspect_processed_data.py`](examples/meta/inspect_processed_data.py) |
| **Results** | Per-config / HPO DataFrames (test error, val error, train time, inference time) | <1 MB | [`run_generate_main_leaderboard.py`](examples/plots/run_generate_main_leaderboard.py) |
| **Leaderboards** | Aggregated ELO, win-rate, average rank, improvability | <1 MB | — |
| **Figures & Plots** | Generated from results and leaderboards | — | — |

</details>


# 📄 Citation

> If you use this code in a scientific publication, please cite the relevant paper(s): **TabArena**
> for the living IID benchmark, and **BeyondArena** for the beyond-IID benchmark.

### TabArena

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

### BeyondArena

**Beyond IID: How General Are Tabular Foundation Models, Really?**
Lennart Purucker, Andrej Tschalzev, Nick Erickson, Gioia Blayer, David Holzmüller, Alan Arazi, Alexander Pfefferle, Mustafa Tajjar, Gaël Varoquaux, Frank Hutter

📄 [arXiv](https://arxiv.org/abs/2606.30410)

<details>
<summary><b>BibTeX</b></summary>

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

</details>


--- 
## Relation to TabRepo 

TabArena was built upon and now replaces [TabRepo](https://arxiv.org/pdf/2311.02971). To see details about TabRepo, the portfolio simulation repository, refer to [tabrepo.md](tabrepo.md).
